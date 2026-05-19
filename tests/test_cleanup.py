from __future__ import annotations

from pathlib import Path

from app.cleanup import run_cleanup_once
from app.repository import RequestRepository


def test_cleanup_deletes_expired_files(tmp_path: Path) -> None:
    db_path = tmp_path / "requests.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "sample.jpg"
    label = uploads / "sample.txt"
    sample.write_bytes(b"image")
    label.write_text("0 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8")

    repo = RequestRepository(db_path)
    repo.init()
    repo.insert_request(
        request_id="req-1",
        stored_path=sample.as_posix(),
        expires_at="2000-01-01T00:00:00Z",
        retention_mode="ttl",
        model_version="test",
        inference_ms=1,
    )

    deleted = run_cleanup_once(repo)
    assert deleted == 1
    assert not sample.exists()
    assert not label.exists()


def test_cleanup_skips_keep_mode_files(tmp_path: Path) -> None:
    db_path = tmp_path / "requests.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "sample.jpg"
    label = uploads / "sample.txt"
    sample.write_bytes(b"image")
    label.write_text("null\n", encoding="utf-8")

    repo = RequestRepository(db_path)
    repo.init()
    repo.insert_request(
        request_id="req-keep",
        stored_path=sample.as_posix(),
        expires_at="9999-12-31T23:59:59Z",
        retention_mode="keep",
        model_version="test",
        inference_ms=1,
    )

    deleted = run_cleanup_once(repo)
    assert deleted == 0
    assert sample.exists()
    assert label.exists()
