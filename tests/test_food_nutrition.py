"""Nutrition arithmetic and provenance checks; fixtures are not food evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from foodcal.nutrition import estimate_calories, load_nutrition


def fixture():
    return {"schema_version": 1, "unit": "kcal_per_100g", "entries": {
        "test_food": {
            "name_zh": "程序测试食物", "status": "verified_reference",
            "kcal_per_100g_low": 120, "kcal_per_100g_high": 180,
            "source_urls": ["https://example.com/test-recipe-a", "https://example.com/test-recipe-b"],
            "references": [
                {"description": "Synthetic recipe A", "kcal_per_100g": 120, "url": "https://example.com/test-recipe-a"},
                {"description": "Synthetic recipe B", "kcal_per_100g": 180, "url": "https://example.com/test-recipe-b"},
            ],
        },
        "unverified": {"name_zh": "待核验", "status": "pending", "kcal_per_100g_low": None,
                       "kcal_per_100g_high": None, "source_urls": [], "references": []},
    }}


class NutritionTests(unittest.TestCase):
    def test_portion_scaling_retains_reference_scope(self):
        result = estimate_calories(fixture(), "test_food", 250)
        self.assertEqual((result["kcal_low"], result["kcal_high"]), (300, 450))
        self.assertEqual(result["mass_source"], "user_input")
        self.assertEqual(result["reference_count"], 2)
        self.assertEqual(result["range_kind"], "listed_reference_foods")

    def test_single_reference_is_not_an_invented_range(self):
        data = fixture()
        row = data["entries"]["test_food"]
        row["references"] = row["references"][:1]
        row["source_urls"] = row["source_urls"][:1]
        row["kcal_per_100g_high"] = 120
        result = estimate_calories(data, "test_food", 125)
        self.assertEqual(result["kcal_low"], result["kcal_high"])
        self.assertEqual(result["kcal_low"], 150)
        self.assertEqual(result["range_kind"], "single_reference_food")

    def test_invalid_masses_and_pending_data_are_rejected(self):
        for value in (0, -1, 5001, float("inf"), float("nan"), True, "100", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_calories(fixture(), "test_food", value)
        for name in ("unverified", "missing", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                estimate_calories(fixture(), name, 100)

    def test_load_rejects_unsourced_or_inconsistent_verified_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nutrition.json"
            path.write_text(json.dumps(fixture()))
            self.assertEqual(load_nutrition(path), fixture())
            changes = [
                {"source_urls": []}, {"references": []}, {"kcal_per_100g_high": 999},
                {"kcal_per_100g_low": float("nan")}, {"status": "automatically_guessed"},
                {"source_urls": ["javascript:alert(1)"]},
            ]
            for change in changes:
                data = fixture()
                data["entries"]["test_food"].update(change)
                path.write_text(json.dumps(data))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_nutrition(path)

    def test_project_table_has_each_configured_class(self):
        root = Path(__file__).resolve().parents[1]
        data = load_nutrition(root / "data/nutrition.json")
        config = json.loads((root / "configs/default.json").read_text())
        self.assertEqual(set(data["entries"]), set(config["classes"]))
        for name, entry in data["entries"].items():
            if entry["status"] == "verified_reference":
                actual = estimate_calories(data, name, 100)
                self.assertEqual(actual["kcal_low"], entry["kcal_per_100g_low"])
                self.assertEqual(actual["kcal_high"], entry["kcal_per_100g_high"])


if __name__ == "__main__":
    unittest.main()
