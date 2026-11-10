"""Verify portable USDA excerpts, reference values, and independent API checks.

Run from the project root with the project's configured Python environment.
This script uses only the files distributed with the repository.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    table_path = ROOT / "data/nutrition.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    source_path = HERE / "usda_energy_records.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    require(source["evidence_kind"] == "selected_record_field_excerpts", "Invalid evidence kind")
    source_hash = sha256(source_path)
    records = {record["food"]["fdcId"]: record["food"] for record in source["records"]}
    require(len(records) == len(source["records"]), "Duplicate source ID")
    checked = []
    by_id = {}
    for class_name, entry in table["entries"].items():
        values = []
        for ref in entry["references"]:
            identifier = ref["fdc_id"]
            require(ref["evidence_file"] == "data/nutrition_sources/usda_energy_records.json", "Nonportable source path")
            require(source_hash == ref["evidence_sha256"], f"Source hash mismatch: {identifier}")
            food = records[identifier]
            require(food["description"] == ref["description"], f"Description mismatch: {identifier}")
            require(food["dataType"] == ref["data_type"], f"Data type mismatch: {identifier}")
            require(food.get("publishedDate") == ref["published_date"], f"Publication date mismatch: {identifier}")
            energies = [nutrient for nutrient in food["foodNutrients"]
                        if nutrient.get("nutrientId") == 1008 and nutrient.get("unitName") == "KCAL"]
            require(len(energies) == 1, f"Missing or ambiguous energy nutrient: {identifier}")
            require(energies[0]["value"] == ref["kcal_per_100g"], f"Energy mismatch: {identifier}")
            require(ref["nutrient_id"] == 1008 and ref["nutrient_unit"] == "KCAL", "Invalid energy identity")
            require(ref["basis_grams"] == 100, "Invalid mass basis")
            require(ref["url"] == f"https://fdc.nal.usda.gov/food-details/{identifier}/nutrients", "Invalid source URL")
            values.append(ref["kcal_per_100g"])
            by_id[identifier] = ref
            checked.append({"class_name": class_name, "fdc_id": identifier, "kcal_per_100g": ref["kcal_per_100g"]})
        require(min(values) == entry["kcal_per_100g_low"], f"Lower bound mismatch: {class_name}")
        require(max(values) == entry["kcal_per_100g_high"], f"Upper bound mismatch: {class_name}")
    require(set(records) == set(by_id), "Unreferenced or missing source records")

    detail_path = HERE / "usda_detail_crosscheck.json"
    details = json.loads(detail_path.read_text(encoding="utf-8"))
    detail_checks = []
    for food in details["records"]:
        identifier = food["fdcId"]
        reference = by_id[identifier]
        energies = [n for n in food["foodNutrients"]
                    if n.get("nutrient", {}).get("id") == 1008
                    and n["nutrient"].get("unitName", "").upper() == "KCAL"]
        require(len(energies) == 1, f"Invalid detail energy: {identifier}")
        require(energies[0]["amount"] == reference["kcal_per_100g"], f"Detail energy mismatch: {identifier}")
        require(food["description"] == reference["description"], f"Detail description mismatch: {identifier}")
        detail_checks.append({"fdc_id": identifier, "kcal_per_100g": energies[0]["amount"], "search_matches_detail": True})
    require({item["fdc_id"] for item in detail_checks} == {170698, 2707060, 2708705, 2708961}, "Unexpected crosscheck set")
    require(len(checked) == 34 and len(table["entries"]) == 10, "Unexpected reference coverage")

    result = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "nutrition_sha256": sha256(table_path),
        "class_count": len(table["entries"]),
        "search_reference_count": len(checked),
        "checks": ["excerpt_file_sha256", "fdc_id", "description", "data_type", "publication_date",
                   "energy_nutrient_1008", "kcal_unit", "per_100g_basis", "official_source_url",
                   "bounds_equal_reference_min_max", "no_unreferenced_source_records"],
        "references": checked,
        "separate_food_detail_crosscheck": detail_checks,
        "evidence_sha256": {source_path.name: source_hash, detail_path.name: sha256(detail_path)},
        "limitations": [
            "Evidence files are unchanged field excerpts, not full HTTP responses.",
            "This verifies local evidence consistency; it does not fetch or authenticate current USDA records.",
            "Selected recipe values are not a statistical confidence interval.",
            "Gyoza and spaghetti_bolognese use explicitly approximate food mappings.",
            "No image classification performance or real meal measurement is evaluated here.",
        ],
    }
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "class_count", "search_reference_count")}, ensure_ascii=False))
    print(f"Separate food-detail crosschecks: {len(detail_checks)} passed")
    return result


if __name__ == "__main__":
    verify()
