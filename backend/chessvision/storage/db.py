"""SQLite storage.

Observations (``analyses.observation_json``), user corrections (``analyses.correction_json``)
and confirmed moves (``moves``) are kept in separate columns/tables so they stay distinguishable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
  id TEXT PRIMARY KEY, filename TEXT, path TEXT NOT NULL, width INTEGER, height INTEGER,
  sha256 TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id),
  calibration_json TEXT NOT NULL, observation_json TEXT NOT NULL, correction_json TEXT,
  mode TEXT NOT NULL, model_json TEXT, preview TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY, title TEXT, start_fen TEXT NOT NULL, start_metadata_json TEXT NOT NULL,
  source TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS moves (
  id TEXT PRIMARY KEY, game_id TEXT NOT NULL REFERENCES games(id), ply INTEGER, uci TEXT NOT NULL,
  provenance TEXT NOT NULL, evidence_json TEXT, t_settled REAL, t_accepted REAL, thumbnail TEXT,
  status TEXT NOT NULL, detached_reason TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS moves_game ON moves(game_id, status, ply);
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY, filename TEXT, path TEXT NOT NULL, meta_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, video_id TEXT NOT NULL REFERENCES videos(id), game_id TEXT REFERENCES games(id),
  status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, media_time REAL NOT NULL DEFAULT 0,
  tracker_state TEXT, params_json TEXT NOT NULL, capabilities_json TEXT, error TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), kind TEXT NOT NULL,
  t_settled REAL, payload_json TEXT NOT NULL, thumbnail TEXT, resolution_json TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON job_events(job_id, created_at);
"""

JSON_SUFFIX = "_json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _encode(values: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for key, value in values.items():
            if key.endswith(JSON_SUFFIX) and value is not None and not isinstance(value, str):
                value = json.dumps(value)
            out[key] = value
        return out

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = {}
        for key in row.keys():
            value = row[key]
            if key.endswith(JSON_SUFFIX):
                out[key[: -len(JSON_SUFFIX)]] = json.loads(value) if value is not None else None
            else:
                out[key] = value
        return out

    def insert(self, table: str, values: dict[str, Any]) -> dict[str, Any]:
        values = dict(values)
        values.setdefault("id", new_id())
        values.setdefault("created_at", now_iso())
        encoded = self._encode(values)
        cols = ", ".join(encoded)
        marks = ", ".join("?" for _ in encoded)
        with self.connect() as conn:
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(encoded.values()))
        return self.get(table, values["id"])  # type: ignore[return-value]

    def get(self, table: str, row_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        return self._decode(row)

    def update(self, table: str, row_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        encoded = self._encode(values)
        sets = ", ".join(f"{k} = ?" for k in encoded)
        with self.connect() as conn:
            conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*encoded.values(), row_id])
        return self.get(table, row_id)

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._decode(r) for r in rows]  # type: ignore[misc]

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    # ---- game helpers -------------------------------------------------
    def active_moves(self, game_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM moves WHERE game_id = ? AND status = 'active' ORDER BY ply", (game_id,)
        )

    def detached_moves(self, game_id: str) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM moves WHERE game_id = ? AND status = 'detached' ORDER BY created_at", (game_id,)
        )
