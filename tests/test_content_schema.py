from __future__ import annotations

import json
from pathlib import Path


V2_CANONICAL_CLASSES = [
    "battery",
    "cable",
    "capacitor",
    "mobile_phone_tablet",
    "printer_multifunction",
    "flat_monitor",
    "mouse",
    "laptop",
    "computer_part",
    "portable_music_player",
    "network_device",
    "landline_telephone",
    "crt_monitor",
    "usb_stick",
    "home_appliance",
    "ink_toner_cartridge",
    "camera",
    "keyboard",
    "power_source_charger",
    "remote",
    "power_tool",
    "clock_radio",
    "home_theater",
    "headset",
    "microphone",
    "smart_watch",
]


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
        assert all(
            "https://" in reference for reference in item["legal_references"]
        ), f"Legal references must include verifiable URLs for {item['class_name']}"

    model_classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert sorted(set(class_names)) == sorted(set(model_classes))
    assert sorted(set(class_names)) == sorted(V2_CANONICAL_CLASSES)


def test_every_v2_class_has_complete_return_card_content() -> None:
    hazards_path = Path(__file__).resolve().parents[1] / "app" / "data" / "hazards_rules.json"
    payload = json.loads(hazards_path.read_text(encoding="utf-8"))
    cards = {item["class_name"]: item for item in payload["classes"]}

    assert sorted(cards) == sorted(V2_CANONICAL_CLASSES)
    for class_name in V2_CANONICAL_CLASSES:
        card = cards[class_name]
        assert card["display_label_pt_br"]
        assert len(card["typical_contents"]) >= 3
        assert card["health_risks"]
        assert card["environmental_risks"]
        assert len(card["disposal_instructions_br"]) >= 3
        assert len(card["legal_references"]) >= 2
