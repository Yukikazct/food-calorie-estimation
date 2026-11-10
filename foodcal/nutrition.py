"""Portion-based estimates from explicitly sourced reference foods.

The bounds describe the listed recipes, not a statistical confidence interval
and not all possible foods in an image category. Images never determine grams.
"""

import json
import math
from pathlib import Path
from urllib.parse import urlparse


MAX_GRAMS = 5000.0


def _finite(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(description + "必须为有限数值")
    return float(value)


def _validate_entry(entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("name_zh"), str):
        raise ValueError("营养条目缺少名称")
    if entry.get("status") not in ("verified_reference", "pending"):
        raise ValueError("营养来源状态无效")
    if entry["status"] == "pending":
        if any(entry.get(key) is not None for key in ("kcal_per_100g_low", "kcal_per_100g_high")):
            raise ValueError("待核验条目不能提供已确认热量值")
        return
    low = _finite(entry.get("kcal_per_100g_low"), "每100克参考值")
    high = _finite(entry.get("kcal_per_100g_high"), "每100克参考值")
    if not 0 < low <= high <= 1000:
        raise ValueError("每100克热量参考范围无效")
    sources = entry.get("source_urls")
    if not isinstance(sources, list) or not sources or any(
        not isinstance(url, str) or urlparse(url).scheme != "https" or not urlparse(url).netloc
        for url in sources
    ):
        raise ValueError("已核验营养条目需要可追溯来源链接")
    references = entry.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("已核验营养条目需要具体参考食物")
    values = []
    for reference in references:
        if (not isinstance(reference, dict) or not isinstance(reference.get("description"), str)
                or not reference["description"].strip() or reference.get("url") not in sources):
            raise ValueError("参考食物缺少说明或来源")
        values.append(_finite(reference.get("kcal_per_100g"), "参考食物热量"))
    if not math.isclose(low, min(values), abs_tol=1e-9) or not math.isclose(high, max(values), abs_tol=1e-9):
        raise ValueError("热量范围与实际参考食物数值不一致")


def load_nutrition(path):
    """Load and validate the local source table; never fetch data at runtime."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or data.get("unit") != "kcal_per_100g" or not isinstance(data.get("entries"), dict)):
        raise ValueError("营养数据格式或单位不正确")
    for name, entry in data["entries"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("营养条目类别名无效")
        _validate_entry(entry)
    return data


def estimate_calories(data, class_name, grams):
    """Scale sourced kcal/100g by an explicitly supplied edible-portion mass."""
    quantity = _finite(grams, "克数")
    if not 0 < quantity <= MAX_GRAMS:
        raise ValueError("请输入大于0且不超过5000克的可食部分重量")
    if not isinstance(data, dict) or data.get("unit") != "kcal_per_100g":
        raise ValueError("营养数据单位不正确")
    entries = data.get("entries")
    if not isinstance(entries, dict) or not isinstance(class_name, str) or class_name not in entries:
        raise ValueError("该菜品没有可用营养条目")
    entry = entries[class_name]
    _validate_entry(entry)
    if entry["status"] != "verified_reference":
        raise ValueError("该菜品的营养来源尚待核验，暂不计算热量")
    return {
        "class_name": class_name,
        "name_zh": entry["name_zh"],
        "grams": quantity,
        "kcal_low": round(entry["kcal_per_100g_low"] * quantity / 100.0, 2),
        "kcal_high": round(entry["kcal_per_100g_high"] * quantity / 100.0, 2),
        "sources": list(entry["source_urls"]),
        "reference_foods": [reference["description"] for reference in entry["references"]],
        "reference_count": len(entry["references"]),
        "range_kind": "listed_reference_foods" if len(entry["references"]) > 1 else "single_reference_food",
        "note": entry.get("note", "按参考配方与输入克数估算；配方、油量及份量误差会影响结果。"),
        "mass_source": "user_input",
    }
