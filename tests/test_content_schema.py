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

V2_25CLASS_MODEL_CLASSES = [
    "battery",
    "cable",
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
    "ink_toner_cartridge",
    "camera",
    "keyboard",
    "power_source_charger",
    "remote",
    "power_tool",
    "clock_radio",
    "headset",
    "microphone",
    "smart_watch",
    "home_appliance",
    "home_theater",
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


def test_v2_25class_runtime_content_matches_model_order() -> None:
    base = Path(__file__).resolve().parents[1] / "app"
    hazards_path = base / "data" / "hazards_rules_v2_25class.json"
    classes_path = base / "model" / "ewaste_v2_25class.classes.txt"

    payload = json.loads(hazards_path.read_text(encoding="utf-8"))
    content_classes = [item["class_name"] for item in payload["classes"]]
    model_classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert model_classes == V2_25CLASS_MODEL_CLASSES
    assert content_classes == V2_25CLASS_MODEL_CLASSES
    assert "capacitor" not in model_classes
    assert "capacitor" not in content_classes


def test_runtime_hazard_content_has_valid_portuguese_encoding() -> None:
    base = Path(__file__).resolve().parents[1] / "app" / "data"
    hazard_paths = [
        base / "hazards_rules.json",
        base / "hazards_rules_v2.json",
        base / "hazards_rules_v2_25class.json",
    ]

    for path in hazard_paths:
        text = path.read_text(encoding="utf-8")
        assert "Ã" not in text
        assert "Â" not in text
        payload = json.loads(text)
        labels = {
            item["class_name"]: item["display_label_pt_br"]
            for item in payload["classes"]
        }
        if "smart_watch" in labels:
            assert labels["smart_watch"] == "relógio inteligente"
