"""Small synthetic files check split governance; they are not food experiments."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import tempfile
import unittest

from foodcal.data import make_manifest


class FoodDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "meta").mkdir()
        self.config = {"classes": ["pizza", "sushi"], "data": {"seed": 42, "validation_fraction": .25}}
        self.official = {"train": {}, "test": {}}
        for c in self.config["classes"]:
            (self.root / "images" / c).mkdir(parents=True)
            for split, count in (("train", 4), ("test", 2)):
                names = [f"{c}/{split}_{i}" for i in range(count)]
                self.official[split][c] = names
                for name in names:
                    (self.root / "images" / (name + ".jpg")).write_bytes(name.encode())
        for split, value in self.official.items():
            (self.root / "meta" / (split + ".json")).write_text(json.dumps(value))

    def test_official_test_preserved_validation_derived_and_repeatable(self):
        first = self.root / "prepared1"
        report = make_manifest(self.root, first, self.config)
        self.assertEqual(report["counts"], {"train": 6, "val": 2, "test": 4})
        with (first / "manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        official_test = {n for values in self.official["test"].values() for n in values}
        self.assertEqual({r["sample_id"] for r in rows if r["split"] == "test"}, official_test)
        self.assertTrue(all(r["source_split"] == "train" for r in rows if r["split"] == "val"))
        self.assertTrue(all(not Path(r["image_path"]).is_absolute() for r in rows))
        second = self.root / "prepared2"
        make_manifest(self.root, second, self.config)
        self.assertEqual((first / "manifest.csv").read_bytes(), (second / "manifest.csv").read_bytes())

    def test_exact_duplicate_groups_are_not_split_and_test_duplicates_exclude_training(self):
        for src, dst in (("pizza/train_0", "pizza/train_1"), ("sushi/test_0", "sushi/train_0")):
            (self.root / "images" / (dst + ".jpg")).write_bytes((self.root / "images" / (src + ".jpg")).read_bytes())
        out = self.root / "prepared"
        report = make_manifest(self.root, out, self.config)
        self.assertEqual(len(report["excluded_training_duplicates_of_test"]), 1)
        with (out / "manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        splits = defaultdict(set)
        for row in rows:
            splits[row["sha256"]].add(row["split"])
        self.assertTrue(all(len(value) == 1 for value in splits.values()))
        self.assertNotIn("sushi/train_0", {r["sample_id"] for r in rows})

    def test_rejects_conflicting_train_labels_and_existing_output(self):
        (self.root / "images/sushi/train_0.jpg").write_bytes((self.root / "images/pizza/train_0.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            make_manifest(self.root, self.root / "prepared", self.config)
        with self.assertRaisesRegex(ValueError, "already exists"):
            make_manifest(self.root, self.root, self.config)
