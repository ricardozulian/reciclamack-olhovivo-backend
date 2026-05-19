from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


class UploadStorage:
    def __init__(self, uploads_dir: Path, retention_hours: int, retention_mode: str = "ttl"):
        self.uploads_dir = uploads_dir
        self.retention_hours = retention_hours
        self.retention_mode = retention_mode
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def save(self, request_id: str, payload: bytes, suffix: str) -> tuple[Path, str]:
        ext = suffix if suffix.startswith(".") else ".bin"
        output = self.uploads_dir / f"{request_id}{ext}"
        output.write_bytes(payload)
        expires_iso = "9999-12-31T23:59:59Z"
        if self.retention_mode == "ttl":
            expires = datetime.utcnow() + timedelta(hours=self.retention_hours)
            expires_iso = expires.replace(microsecond=0).isoformat() + "Z"
        return output, expires_iso

    def save_label(self, image_path: Path, content: str) -> Path:
        output = image_path.with_suffix(".txt")
        output.write_text(content.rstrip() + "\n", encoding="utf-8")
        return output
