"""Fixed-split food training. Validation chooses weights; testing is separate."""

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import classification_metrics
from .model import ARCHITECTURE, NORMALIZATION, build_model, image_transform, load_classifier, resolve_device, validate_classes


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = ("sample_id", "image_path", "class_name", "split", "source_split", "sha256")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_manifest(manifest, class_names, *, verify_splits=()):
    """Audit all split metadata, but open image bytes only in requested splits."""
    classes = validate_classes(class_names)
    path = Path(manifest).resolve()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (not reader.fieldnames or not set(MANIFEST_FIELDS) <= set(reader.fieldnames)
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError("Missing or duplicated manifest columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    ids, hashes, paths = set(), {}, {}
    for row in rows:
        if (None in row or any(row.get(key) is None or not row[key].strip() for key in MANIFEST_FIELDS)
                or row["sample_id"] in ids):
            raise ValueError("Malformed manifest or duplicate sample_id")
        ids.add(row["sample_id"])
        split = row["split"]
        if split not in ("train", "val", "test") or row["class_name"] not in classes:
            raise ValueError("Manifest split or class is not in the configured vocabulary")
        if row["source_split"] != ("test" if split == "test" else "train"):
            raise ValueError("Preserve official test; validation must come only from official train")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Every image requires its lowercase SHA-256")
        relative = Path(row["image_path"])
        if relative.is_absolute():
            raise ValueError("image_path must be relative to the manifest directory")
        resolved = (path.parent / relative).resolve()
        for ledger, key in ((hashes, row["sha256"]), (paths, str(resolved))):
            if ledger.setdefault(key, split) != split:
                raise ValueError("An identical image or source path crosses dataset splits")
        row["_path"] = resolved
        row["_label"] = classes.index(row["class_name"])
        if split in verify_splits:
            if not resolved.is_file() or _sha(resolved) != row["sha256"]:
                raise ValueError(f"Missing or changed image: {row['sample_id']}")
            with Image.open(resolved) as image:
                image.verify()
    return rows


class FoodDataset(Dataset):
    def __init__(self, rows, input_size=224, augment=False):
        self.rows = rows
        self.transform = image_transform(input_size, augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["_path"]) as image:
            tensor = self.transform(image)
        return tensor, row["_label"]


def configure_trainable(model, mode):
    if mode not in ("frozen", "last_block"):
        raise ValueError("mode must be frozen or last_block")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    if mode == "last_block":
        for parameter in model.features[-1].parameters():
            parameter.requires_grad_(True)


def set_training_mode(model, mode):
    if mode not in ("frozen", "last_block"):
        raise ValueError("mode must be frozen or last_block")
    model.train()
    model.features.eval()  # Frozen BatchNorm statistics must remain frozen too.
    if mode == "last_block":
        model.features[-1].train()


def _seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(rows, input_size, augment, settings, *, shuffle=False):
    generator = torch.Generator().manual_seed(settings["seed"])
    return DataLoader(FoodDataset(rows, input_size, augment), batch_size=settings["batch_size"],
                      shuffle=shuffle, num_workers=settings["num_workers"], generator=generator,
                      worker_init_fn=_seed_worker, drop_last=False)


def _settings(config):
    values = {"seed": 42, "batch_size": 32, "epochs": 12, "patience": 4,
              "learning_rate": .001, "backbone_learning_rate": .0001, "num_workers": 0,
              "weight_decay": .0001}
    values.update(config.get("training", {}))
    for key in ("batch_size", "epochs", "patience"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    for key in ("seed", "num_workers"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError(key + " must be a nonnegative integer")
    for key in ("learning_rate", "backbone_learning_rate", "weight_decay"):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "weight_decay" and not value):
            raise ValueError(key + " must be a valid optimizer rate")
    return values


def _config(value):
    if isinstance(value, dict):
        config = copy.deepcopy(value)
    else:
        config = json.loads(Path(value).read_text(encoding="utf-8"))
    classes = validate_classes(config.get("classes"))
    translated = config.get("class_names_zh", list(classes))
    if not isinstance(translated, list) or len(translated) != len(classes) or any(not isinstance(x, str) or not x for x in translated):
        raise ValueError("class_names_zh must follow the configured class order")
    config["class_names_zh"] = translated
    if config.get("model", {}).get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("Only mobilenet_v3_small is supported")
    return config


def _infer(model, loader, device, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    probabilities, labels, total_loss = [], [], 0.
    start = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device))
            total_loss += float(criterion(logits, target.to(device)).cpu())
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            labels.append(target.numpy())
    if not labels:
        raise ValueError("The requested evaluation split is empty")
    y, p = np.concatenate(labels), np.concatenate(probabilities)
    metrics = classification_metrics(y, p.argmax(axis=1), class_names)
    metrics["cross_entropy"] = total_loss / len(y)
    metrics["evaluation_seconds"] = time.perf_counter() - start
    metrics["timing_scope"] = "Batch evaluation including image loading/transforms, model forward, loss and CPU result transfer; not web or camera FPS"
    return metrics, p


def _save_predictions(output, rows, probabilities, metrics, class_names):
    _write_json(output / "metrics.json", metrics)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["sample_id", "reference", "prediction", "confidence"] + ["p_" + name for name in class_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, vector in zip(rows, probabilities):
            predicted = int(vector.argmax())
            writer.writerow({"sample_id": row["sample_id"], "reference": row["class_name"],
                             "prediction": class_names[predicted], "confidence": float(vector[predicted]),
                             **{"p_" + name: float(value) for name, value in zip(class_names, vector)}})
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reference/prediction"] + list(class_names))
        for name, values in zip(class_names, metrics["confusion_matrix"]):
            writer.writerow([name] + values)


def _plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [row["epoch"] for row in history]
    for name in ("train_loss", "val_loss"):
        axes[0].plot(epochs, [row[name] for row in history], label=name)
    for name in ("val_accuracy", "val_macro_f1"):
        axes[1].plot(epochs, [row[name] for row in history], label=name)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Cross entropy")
    axes[1].set_ylabel("Validation score")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=150)
    plt.close(figure)


def train(manifest, config, output, mode="frozen", augment=False, device="auto"):
    config = _config(config)
    settings = _settings(config)
    classes = tuple(config["classes"])
    input_size = config.get("model", {}).get("input_size", 224)
    image_transform(input_size, augment)
    if type(augment) is not bool:
        raise ValueError("augment must be a boolean")
    if mode not in ("frozen", "last_block"):
        raise ValueError("mode must be frozen or last_block")
    pretrained = config.get("model", {}).get("pretrained_weights")
    if not pretrained:
        raise ValueError("Training requires a supplied local pretrained ImageNet checkpoint")
    weights = Path(pretrained)
    if not weights.is_absolute():
        weights = ROOT / weights
    target_device = resolve_device(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    try:
        rows = read_manifest(manifest, classes, verify_splits=("train", "val"))
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        counts = {split: {name: sum(row["class_name"] == name for row in selected) for name in classes}
                  for split, selected in (("train", train_rows), ("val", val_rows))}
        if any(count == 0 for group in counts.values() for count in group.values()):
            raise ValueError("Every configured class needs at least one training and validation image")
        seed = settings["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(len(classes), weights).to(target_device)
        configure_trainable(model, mode)
        groups = [{"params": list(model.classifier.parameters()), "lr": settings["learning_rate"]}]
        if mode == "last_block":
            groups.append({"params": list(model.features[-1].parameters()), "lr": settings["backbone_learning_rate"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        train_loader = _loader(train_rows, input_size, augment, settings, shuffle=True)
        val_loader = _loader(val_rows, input_size, False, settings)
        protocol = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
            "class_names": list(classes), "training_settings": settings,
            "mode": mode, "augment": augment, "device": str(target_device),
            "augmentation": "ColorJitter(.2,.2,.2,.02) only" if augment else "none",
            "preprocessing": "EXIF-correct RGB, bilinear resize256/224 ratio, center crop, ImageNet normalization",
            "trainable_scope": "classifier only" if mode == "frozen" else "features[-1] and classifier only",
            "batch_norm": "Frozen feature modules remain eval, including running statistics",
            "selection": "Earliest maximum validation macro-F1; test images never loaded during training",
            "manifest_sha256": _sha(manifest), "pretrained_sha256": _sha(weights),
            "implementation_sha256": {name: _sha(Path(__file__).parent / name)
                                      for name in ("model.py", "training.py", "metrics.py")},
            "counts": counts, "test_evaluated": False,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
        _write_json(out / "protocol.json", protocol)
        best_score, best_epoch, best_state = -1., 0, None
        history = []
        started = time.perf_counter()
        for epoch in range(1, settings["epochs"] + 1):
            set_training_mode(model, mode)
            total_loss, seen, correct = 0., 0, 0
            for images, target in train_loader:
                images, target = images.to(target_device), target.to(target_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(target)
                correct += int((logits.detach().argmax(1) == target).sum().cpu())
                seen += len(target)
            validation, _ = _infer(model, val_loader, target_device, classes)
            row = {"epoch": epoch, "train_loss": total_loss / seen, "train_accuracy": correct / seen,
                   "val_loss": validation["cross_entropy"], "val_accuracy": validation["accuracy"],
                   "val_macro_f1": validation["macro_f1"]}
            history.append(row)
            if row["val_macro_f1"] > best_score + 1e-12:
                best_score, best_epoch = row["val_macro_f1"], epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                checkpoint = {"format": "foodcal-classifier", "version": 1, "trained": True,
                              "architecture": ARCHITECTURE, "state_dict": best_state,
                              "class_names": list(classes), "class_names_zh": config["class_names_zh"],
                              "input_size": input_size, "normalization": NORMALIZATION,
                              "best_epoch": best_epoch, "mode": mode, "augment": augment, "seed": seed,
                              "manifest_sha256": protocol["manifest_sha256"],
                              "pretrained_sha256": protocol["pretrained_sha256"],
                              "training_settings": settings, "data": config.get("data", {}),
                              "selection": protocol["selection"], "validation_macro_f1": best_score,
                              "test_evaluated": False}
                torch.save(checkpoint, out / "checkpoint.pt")
            with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps(row), flush=True)
            if epoch - best_epoch >= settings["patience"]:
                break
        model.load_state_dict(best_state)
        validation, probabilities = _infer(model, val_loader, target_device, classes)
        validation.update(split="val", purpose="Development validation used for checkpoint selection; not final test")
        _save_predictions(out, val_rows, probabilities, validation, classes)
        _plot_history(history, out)
        summary = {"best_epoch": best_epoch, "epochs_completed": len(history),
                   "elapsed_seconds": time.perf_counter() - started,
                   "accuracy": validation["accuracy"], "macro_f1": validation["macro_f1"],
                   "per_class_recall": validation["per_class_recall"],
                   "checkpoint": str(out / "checkpoint.pt"), "class_names": list(classes),
                   "mode": mode, "augment": augment, "seed": seed, "device": str(target_device),
                   "total_parameters": protocol["total_parameters"], "trainable_parameters": protocol["trainable_parameters"],
                   "manifest_sha256": protocol["manifest_sha256"], "test_evaluated": False,
                   "data_scope": config.get("data", {}).get("dataset", "user-supplied manifest"),
                   "metric_scope": "Validation-selected development result"}
        _write_json(out / "summary.json", summary)
        return summary
    except Exception as exc:
        _write_json(out / "failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                                           "scope": "Failed attempt; no completed experiment claim"})
        raise


def evaluate(manifest, checkpoint, output, split="test", device="auto"):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    target_device = resolve_device(device)
    model, metadata = load_classifier(checkpoint, target_device)
    classes = tuple(metadata["class_names"])
    rows = read_manifest(manifest, classes, verify_splits=(split,))
    if _sha(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Evaluation requires the frozen training manifest; use a separate external-data protocol for a new manifest")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError("Requested evaluation split is empty")
    settings = _settings({"training": metadata["training_settings"]})
    loader = _loader(selected, metadata["input_size"], False, settings)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    try:
        metrics, probabilities = _infer(model, loader, target_device, classes)
        metrics.update(split=split, checkpoint_sha256=_sha(checkpoint), manifest_sha256=_sha(manifest),
                       device=str(target_device), checkpoint_best_epoch=metadata["best_epoch"],
                       purpose="Final held-out test; do not use to choose or tune models" if split == "test" else "Development validation")
        _save_predictions(out, selected, probabilities, metrics, classes)
        return metrics
    except Exception as exc:
        _write_json(out / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
