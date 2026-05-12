from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


class UploadStorage:
    def __init__(self, uploads_dir: Path, retention_hours: int):
        self.uploads_dir = uploads_dir
        self.retention_hours = retention_hours
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def save(self, request_id: str, payload: bytes, suffix: str) -> tuple[Path, str]:
        ext = suffix if suffix.startswith(".") else ".bin"
        output = self.uploads_dir / f"{request_id}{ext}"
        output.write_bytes(payload)
        expires = datetime.utcnow() + timedelta(hours=self.retention_hours)
        expires_iso = expires.replace(microsecond=0).isoformat() + "Z"
        return output, expires_iso

