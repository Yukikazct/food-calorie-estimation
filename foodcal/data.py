"""Prepare a fixed Food-101 subset without using official test data for fitting."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import tarfile


ARCHIVE_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
ARCHIVE_MD5 = "85eeb15f3717b99a5da872d97d918f87"
ARCHIVE_BYTES = 4996278331
FIELDS = ["sample_id", "image_path", "class_name", "split", "source_split", "sha256"]


def load_config(path="configs/default.json"):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = value["classes"]
    if not isinstance(classes, list) or len(classes) != len(set(classes)) or len(classes) < 2:
        raise ValueError("Expected distinct food classes")
    if len(value["class_names_zh"]) != len(classes):
        raise ValueError("Class labels disagree")
    if any(not c or PurePosixPath(c).name != c or c in (".", "..") for c in classes):
        raise ValueError("Invalid class name")
    return value


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_subset(archive, image_root, classes):
    archive, image_root = Path(archive), Path(image_root)
    if file_hash(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Archive checksum does not match the official Food-101 release")
    image_root.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            parts = PurePosixPath(member.name).parts
            if (not member.isfile() or len(parts) < 2 or parts[0] != "food-101"
                    or any(p in ("..", ".") for p in parts)):
                continue
            wanted_image = len(parts) == 4 and parts[1] == "images" and parts[2] in classes and parts[3].endswith(".jpg")
            wanted_meta = len(parts) == 3 and parts[1] == "meta" and parts[2] in ("train.json", "test.json", "classes.txt", "labels.txt")
            wanted_readme = len(parts) == 2 and parts[1].lower() in ("readme.txt", "readme.md", "license", "license.txt")
            if not (wanted_image or wanted_meta or wanted_readme):
                continue
            destination = image_root.joinpath(*parts[1:])
            if not destination.resolve().is_relative_to(image_root.resolve()):
                raise ValueError("Unsafe archive member")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, destination.open("wb") as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(chunk)
            count += int(wanted_image)
    return count


def make_manifest(image_root, output, config):
    """Retain official test; validation is class-stratified by exact-image group."""
    config = load_config(config) if isinstance(config, (str, Path)) else config
    image_root, output = Path(image_root).resolve(), Path(output)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new manifest version")
    official = {split: json.loads((image_root / "meta" / (split + ".json")).read_text())
                for split in ("train", "test")}
    classes = config["classes"]
    fraction = float(config["data"]["validation_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    records = []
    seen_ids = set()
    for source_split in ("train", "test"):
        for cls in classes:
            if cls not in official[source_split]:
                raise ValueError("Food-101 class unavailable: " + cls)
            for name in official[source_split][cls]:
                if PurePosixPath(name).parts[0] != cls or len(PurePosixPath(name).parts) != 2:
                    raise ValueError("Invalid official sample identifier")
                if name in seen_ids:
                    raise ValueError("Official sample ID crosses source splits")
                seen_ids.add(name)
                path = image_root / "images" / (name + ".jpg")
                if not path.is_file():
                    raise ValueError("Missing source image: " + str(path))
                records.append({"sample_id": name, "image_path": str(path), "class_name": cls,
                                "source_split": source_split, "split": source_split, "sha256": file_hash(path)})
    test_hashes = {r["sha256"] for r in records if r["source_split"] == "test"}
    excluded = [r for r in records if r["source_split"] == "train" and r["sha256"] in test_hashes]
    records = [r for r in records if not (r["source_split"] == "train" and r["sha256"] in test_hashes)]
    training_groups = defaultdict(list)
    for row in records:
        if row["source_split"] == "train":
            training_groups[row["sha256"]].append(row)
    if any(len({r["class_name"] for r in group}) > 1 for group in training_groups.values()):
        raise ValueError("Identical training images have conflicting class labels; review before preparing")
    rng = random.Random(config["data"]["seed"])
    for cls in classes:
        groups = sorted(key for key, group in training_groups.items() if group[0]["class_name"] == cls)
        if len(groups) < 2:
            raise ValueError("Need at least two independent training image groups per class")
        rng.shuffle(groups)
        count = min(len(groups)-1, max(1, round(len(groups)*fraction)))
        for key in groups[:count]:
            for row in training_groups[key]:
                row["split"] = "val"
    output.mkdir(parents=True)
    for row in records:
        row["image_path"] = os.path.relpath(row["image_path"], output.resolve())
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    report = {"dataset": "Food-101", "classes": classes, "seed": config["data"]["seed"],
              "counts": dict(Counter(r["split"] for r in records)),
              "per_class": {c: dict(Counter(r["split"] for r in records if r["class_name"] == c)) for c in classes},
              "validation_policy": f"{fraction:.0%} of official train exact-image groups per class; no official test used",
              "duplicate_policy": "Keep official test intact; exclude exact test duplicates from official train; keep train/val hash groups together",
              "excluded_training_duplicates_of_test": excluded, "manifest_sha256": file_hash(manifest),
              "official_metadata_sha256": {s: file_hash(image_root / "meta" / (s + ".json")) for s in official},
              "license": "Images remain property of original owners; Food-101 authors describe use beyond scientific fair use as requiring negotiation with owners",
              "source_url": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
              "training_completed": False}
    (output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
