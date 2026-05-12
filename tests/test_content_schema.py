from __future__ import annotations

import json
from pathlib import Path


def test_hazards_entries_have_legal_references() -> None:
    base = Path(__file__).resolve().parents[1] / "app"
    hazards_path = base / "data" / "hazards_rules.json"
    classes_path = base / "model" / "yolo11n_ewaste.classes.txt"

    payload = json.loads(hazards_path.read_text(encoding="utf-8"))
    classes = payload["classes"]

    assert classes, "hazards_rules.json must include at least one class"
    class_names = []
    for item in classes:
        class_names.append(item["class_name"])
        assert item["class_name"]
        assert item["display_label_pt_br"]
        assert item["hazard_level"]
        assert item["disposal_instructions_br"]
        assert item["legal_references"], f"Missing legal references for {item['class_name']}"

    model_classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert sorted(set(class_names)) == sorted(set(model_classes))
