"""Mocked orchestration checks; no training or food-recognition results."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_food_experiments.py"
SPEC = importlib.util.spec_from_file_location("food_suite_under_test", SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


class FoodSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "runs/suite"
        (self.root / "foodcal").mkdir()
        for name in ("model.py", "training.py", "metrics.py"):
            (self.root / "foodcal" / name).write_text("# Mock implementation fingerprint\n")
        self.weights = self.root / "initialization.bin"
        self.weights.write_bytes(b"mock initialization; never loaded as a model")
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("mock manifest; never used for training\n")
        self.config = {
            "classes": ["pizza", "sushi"], "class_names_zh": ["披萨", "寿司"],
            "data": {"manifest": str(self.manifest)},
            "model": {"pretrained_weights": str(self.weights)},
        }
        self.config_path = self.root / "config.json"
        self.write_json(self.config_path, self.config)
        self.events = []
        # B has the best validation Macro-F1; A leads on validation accuracy
        # and test Macro-F1. These arbitrary values test selection, not accuracy.
        self.val_f1 = {"A_frozen": .8, "B_last_block": .9, "C_last_block_color": .7}
        self.val_accuracy = {"A_frozen": .99, "B_last_block": .8, "C_last_block_color": .9}
        self.test_f1 = {"A_frozen": .99, "B_last_block": .5, "C_last_block_color": .7}

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def read_json(self, relative):
        return json.loads((self.output / relative).read_text())

    def fake_train(self, manifest, config, output, mode, augment, device):
        self.assertEqual(Path(manifest), self.manifest)
        self.assertEqual(config, self.config)
        name = output.name
        self.events.append(("train", name))
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoint.pt").write_bytes(("mock checkpoint " + name).encode())
        protocol = {
            "config": config, "pretrained_sha256": suite.file_hash(self.weights),
            "implementation_sha256": {key: suite.file_hash(self.root / "foodcal" / key)
                                      for key in ("model.py", "training.py", "metrics.py")},
        }
        summary = {
            "manifest_sha256": suite.file_hash(self.manifest), "mode": mode, "augment": augment,
            "macro_f1": self.val_f1[name], "accuracy": self.val_accuracy[name],
            "best_epoch": 1, "epochs_completed": 1, "elapsed_seconds": 0.,
        }
        self.write_json(output / "protocol.json", protocol)
        self.write_json(output / "summary.json", summary)
        return summary

    def fake_evaluate(self, manifest, checkpoint, output, split, device):
        self.assertEqual(split, "test")
        self.assertTrue(all((self.output / name / "summary.json").is_file() for name in self.val_f1))
        # Read the actual persisted selection inside every evaluation call.
        selection = self.read_json("selection.json")
        self.assertEqual(selection["condition"], "B_last_block")
        self.assertEqual(selection["basis"], "validation_macro_f1")
        self.assertFalse(selection["test_used_for_selection"])
        name = checkpoint.parent.name
        self.events.append(("evaluate", name))
        output.mkdir(parents=True, exist_ok=False)
        metrics = {
            "checkpoint_sha256": suite.file_hash(checkpoint),
            "manifest_sha256": suite.file_hash(manifest), "split": "test",
            "accuracy": self.test_f1[name], "macro_f1": self.test_f1[name],
        }
        self.write_json(output / "metrics.json", metrics)
        (output / "predictions.csv").write_text("mock,prediction\n")
        (output / "confusion_matrix.csv").write_text("mock,confusion\n")
        return metrics

    def run_suite(self, train=None, evaluate=None):
        original_cwd = Path.cwd()
        try:
            with patch.object(suite, "ROOT", self.root), patch.object(
                sys, "argv", [str(SCRIPT), "--config", str(self.config_path),
                              "--output", str(self.output), "--device", "cpu"]
            ), patch.object(suite, "train", side_effect=train or self.fake_train) as train_call, patch.object(
                suite, "evaluate", side_effect=evaluate or self.fake_evaluate
            ) as evaluate_call, patch(
                "scripts.check_food_model.check",
                side_effect=lambda config, output, device: self.write_json(output, {"scope": "Mock preflight"})
            ), redirect_stdout(io.StringIO()):
                suite.main()
                return train_call, evaluate_call
        finally:
            os.chdir(original_cwd)

    def test_selection_is_persisted_before_test_and_uses_only_validation_macro_f1(self):
        train_call, evaluate_call = self.run_suite()
        self.assertEqual(train_call.call_count, 3)
        self.assertEqual(evaluate_call.call_count, 3)
        self.assertEqual([kind for kind, _ in self.events], ["train"] * 3 + ["evaluate"] * 3)
        comparison = self.read_json("comparison.json")
        self.assertEqual(comparison["selection"]["condition"], "B_last_block")
        self.assertEqual(max(comparison["results"], key=lambda row: row["test_macro_f1"])["condition"], "A_frozen")
        self.assertEqual((self.root / "models/food10/best.pt").read_bytes(),
                         (self.output / "B_last_block/checkpoint.pt").read_bytes())

    def test_resume_skips_complete_conditions_and_complete_evaluations(self):
        def stop_before_second_condition(*args):
            if args[2].name == "B_last_block":
                raise RuntimeError("Mock interruption between complete conditions")
            return self.fake_train(*args)

        with self.assertRaisesRegex(RuntimeError, "Mock interruption"):
            self.run_suite(train=stop_before_second_condition)
        self.assertEqual(self.read_json("status.json")["completed"], ["A_frozen"])
        self.events.clear()
        train_call, evaluate_call = self.run_suite()
        self.assertEqual(train_call.call_count, 2)
        self.assertEqual(evaluate_call.call_count, 3)
        self.assertNotIn(("train", "A_frozen"), self.events)
        train_call, evaluate_call = self.run_suite(
            train=AssertionError("Completed training must not repeat"),
            evaluate=AssertionError("Completed testing must not repeat"))
        train_call.assert_not_called()
        evaluate_call.assert_not_called()
        self.assertEqual(self.read_json("status.json")["state"], "completed")

    def test_metrics_without_completion_marker_is_rejected_on_resume(self):
        def interrupted_evaluation(manifest, checkpoint, output, split, device):
            self.assertEqual(self.read_json("selection.json")["condition"], "B_last_block")
            output.mkdir(parents=True)
            self.write_json(output / "metrics.json", {"scope": "Mock partial output"})
            raise RuntimeError("Mock interruption while writing evaluation artifacts")

        with self.assertRaisesRegex(RuntimeError, "Mock interruption"):
            self.run_suite(evaluate=interrupted_evaluation)
        self.assertFalse((self.output / "A_frozen_test/completion.json").exists())
        with self.assertRaisesRegex(ValueError, "Incomplete evaluation preserved"):
            self.run_suite(train=AssertionError("Completed training must not repeat"),
                           evaluate=AssertionError("Incomplete testing must not silently repeat"))
        self.assertEqual(self.read_json("status.json")["state"], "failed")
        self.assertFalse((self.root / "models/food10/best.pt").exists())

    def test_changed_completed_evaluation_artifact_is_rejected(self):
        self.run_suite()
        (self.output / "A_frozen_test/predictions.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Completed evaluation artifacts have changed"):
            self.run_suite(train=AssertionError("Completed training must not repeat"),
                           evaluate=AssertionError("Changed testing must not silently repeat"))
        self.assertEqual(self.read_json("status.json")["state"], "failed")


if __name__ == "__main__":
    unittest.main()
