from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class RequestRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS request_logs (
                  request_id TEXT PRIMARY KEY,
                  stored_path TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  deleted_at TEXT,
                  status TEXT NOT NULL,
                  model_version TEXT NOT NULL,
                  inference_ms INTEGER NOT NULL
                );
                """
            )
            conn.commit()

    def insert_request(
        self,
        request_id: str,
        stored_path: str,
        expires_at: str,
        model_version: str,
        inference_ms: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO request_logs (
                  request_id, stored_path, created_at, expires_at, deleted_at, status, model_version, inference_ms
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    request_id,
                    stored_path,
                    utc_now_iso(),
                    expires_at,
                    "stored",
                    model_version,
                    inference_ms,
                ),
            )
            conn.commit()

    def mark_deleted(self, request_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE request_logs
                   SET deleted_at = ?, status = ?
                 WHERE request_id = ?
                """,
                (utc_now_iso(), "deleted", request_id),
            )
            conn.commit()

    def fetch_expired(self, now_iso: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT request_id, stored_path
                  FROM request_logs
                 WHERE status = 'stored'
                   AND expires_at <= ?
                """,
                (now_iso,),
            ).fetchall()
            return rows

