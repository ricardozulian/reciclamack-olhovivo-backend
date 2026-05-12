from __future__ import annotations

from pathlib import Path

from app.cleanup import run_cleanup_once
from app.repository import RequestRepository


def test_cleanup_deletes_expired_files(tmp_path: Path) -> None:
    db_path = tmp_path / "requests.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "sample.jpg"
    sample.write_bytes(b"image")

    repo = RequestRepository(db_path)
    repo.init()
    repo.insert_request(
        request_id="req-1",
        stored_path=sample.as_posix(),
        expires_at="2000-01-01T00:00:00Z",
        model_version="test",
        inference_ms=1,
    )

    deleted = run_cleanup_once(repo)
    assert deleted == 1
    assert not sample.exists()

