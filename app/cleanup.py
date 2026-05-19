from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from .repository import RequestRepository


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def run_cleanup_once(repository: RequestRepository) -> int:
    expired = repository.fetch_expired(now_iso())
    deleted = 0
    for row in expired:
        path = Path(row["stored_path"])
        label_path = path.with_suffix(".txt")
        if path.exists():
            path.unlink(missing_ok=True)
        if label_path.exists():
            label_path.unlink(missing_ok=True)
        repository.mark_deleted(row["request_id"])
        deleted += 1
    return deleted


async def cleanup_loop(repository: RequestRepository, interval_seconds: int) -> None:
    while True:
        run_cleanup_once(repository)
        await asyncio.sleep(interval_seconds)
