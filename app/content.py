from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContentStore:
    def __init__(self, path: Path):
        self.path = path
        self.content_version = "unknown"
        self._class_map: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.content_version = raw.get("content_version", "unknown")
        self._class_map = {
            item["class_name"].lower(): item for item in raw.get("classes", []) if item.get("class_name")
        }

    def supported_classes(self) -> list[str]:
        return sorted(self._class_map.keys())

    def get_guidance(self, class_name: str) -> dict[str, Any] | None:
        return self._class_map.get(class_name.lower())

    def get_display_label(self, class_name: str) -> str:
        info = self.get_guidance(class_name)
        if not info:
            return class_name
        return str(info.get("display_label_pt_br") or class_name)
