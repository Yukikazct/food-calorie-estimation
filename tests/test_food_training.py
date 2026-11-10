"""Software checks use synthetic inputs; none are food-recognition results."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from foodcal.metrics import classification_metrics
from foodcal.model import build_model, image_transform, load_classifier
from foodcal.training import configure_trainable, read_manifest, set_training_mode, train, evaluate


ROOT = Path(__file__).resolve().parents[1]


class FoodTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_metrics_from_hand_counted_confusion_and_absent_class(self):
        result = classification_metrics([0, 0, 1, 2], [0, 1, 1, 0], ["a", "b", "c"])
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 7 / 18)
        self.assertEqual(result["per_class_recall"], [.5, 1., 0.])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
        self.assertIsNone(classification_metrics([0], [0], ["a", "b"])["per_class_recall"][1])
        for truth, prediction in (([], []), ([0], [-1]), ([0], [2]), ([0.5], [0])):
            with self.assertRaises(ValueError):
                classification_metrics(truth, prediction, ["a", "b"])

    def test_frozen_features_and_batchnorm_remain_bit_identical_after_step(self):
        torch.manual_seed(9)
        model = build_model(3)
        configure_trainable(model, "frozen")
        set_training_mode(model, "frozen")
        before = {name: value.clone() for name, value in model.features.state_dict().items()}
        head = model.classifier[-1].weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 1])).backward()
        optimizer.step()
        for name, value in model.features.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(torch.equal(head, model.classifier[-1].weight))
        self.assertFalse(model.features.training)
        self.assertTrue(model.classifier.training)

    def test_last_block_control_only_unfreezes_declared_scope(self):
        model = build_model(3)
        configure_trainable(model, "last_block")
        set_training_mode(model, "last_block")
        last = len(model.features) - 1
        for name, parameter in model.named_parameters():
            expected = name.startswith("classifier.") or name.startswith(f"features.{last}.")
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertFalse(model.features[0].training)
        self.assertTrue(model.features[-1].training)

    def _manifest(self, directory):
        rows = []
        for split in ("train", "val"):
            for label in range(2):
                for index in range(2):
                    sample = f"{split}_{label}_{index}"
                    pixels = np.random.default_rng(len(rows)).integers(0, 256, (48, 64, 3), dtype=np.uint8)
                    path = directory / (sample + ".png")
                    Image.fromarray(pixels).save(path)
                    rows.append({"sample_id": sample, "image_path": path.name, "class_name": ["pizza", "sushi"][label],
                                 "split": split, "source_split": "train", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.append({"sample_id": "not_opened_test", "image_path": "missing_test_image.png", "class_name": "sushi",
                     "split": "test", "source_split": "test", "sha256": "a" * 64})
        path = directory / "manifest.csv"
        self._write_manifest(path, rows)
        return path, rows

    @staticmethod
    def _write_manifest(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_manifest_official_split_duplicate_hash_and_changed_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, rows = self._manifest(root)
            self.assertEqual(len(read_manifest(path, ["pizza", "sushi"], verify_splits=("train", "val"))), 9)
            original = [dict(row) for row in rows]
            for change in ("hash", "official", "sample_id", "actual_bytes"):
                rows = [dict(row) for row in original]
                if change == "hash":
                    rows[4]["sha256"] = rows[0]["sha256"]
                elif change == "official":
                    rows[4]["source_split"] = "test"
                elif change == "sample_id":
                    rows[4]["sample_id"] = rows[0]["sample_id"]
                else:
                    rows[0]["sha256"] = "b" * 64
                self._write_manifest(path, rows)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_manifest(path, ["pizza", "sushi"], verify_splits=("train", "val"))

    def test_training_round_trip_never_opens_test_and_rejects_imagenet_as_food(self):
        weights = ROOT / "models/pretrained/mobilenet_v3_small-047dcff4.pth"
        if not weights.is_file():
            self.skipTest("Local ImageNet checkpoint not present")
        with self.assertRaisesRegex(ValueError, "trained foodcal"):
            load_classifier(weights)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, rows = self._manifest(root)
            config = {"classes": ["pizza", "sushi"], "class_names_zh": ["披萨", "寿司"],
                      "data": {"dataset": "synthetic software check, not food accuracy"},
                      "model": {"pretrained_weights": str(weights), "input_size": 32},
                      "training": {"epochs": 1, "patience": 1, "batch_size": 2, "num_workers": 0}}
            summary = train(path, config, root / "run", device="cpu")
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(summary["epochs_completed"], 1)
            model, metadata = load_classifier(root / "run/checkpoint.pt")
            self.assertEqual(metadata["class_names"], ["pizza", "sushi"])
            self.assertEqual(model(torch.randn(1, 3, 32, 32)).shape, (1, 2))
            result = evaluate(path, root / "run/checkpoint.pt", root / "val", split="val", device="cpu")
            self.assertEqual(result["sample_count"], 4)
            self.assertAlmostEqual(result["macro_f1"], summary["macro_f1"])
            with self.assertRaisesRegex(ValueError, "Missing or changed image"):
                evaluate(path, root / "run/checkpoint.pt", root / "test", device="cpu")
            with self.assertRaises(FileExistsError):
                train(path, config, root / "run", device="cpu")
            self.assertTrue((root / "run/training_curves.png").is_file())
            protocol = json.loads((root / "run/protocol.json").read_text())
            self.assertNotIn("test", protocol["counts"])

    def test_preprocessing_has_same_shape_and_normalization_for_rgba_inputs(self):
        image = Image.new("RGBA", (100, 50), (120, 90, 60, 150))
        actual = image_transform(32)(image)
        self.assertEqual(actual.shape, (3, 32, 32))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(image_transform(32, True)(image).shape, actual.shape)


if __name__ == "__main__":
    unittest.main()
