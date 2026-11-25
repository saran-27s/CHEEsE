from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import HTTPException, Request

from ..config import Settings
from ..inference.registry import ModelRegistry
from ..jobs.video_job import VideoJobRunner
from ..jobs.worker import JobWorker
from ..storage.db import Database

SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class AppState:
    settings: Settings
    db: Database
    registry: ModelRegistry
    worker: JobWorker
    video_runner: VideoJobRunner


def get_state(request: Request) -> AppState:
    return request.app.state.cv


def file_url(kind: str, name: str | None) -> str | None:
    return f"/api/files/{kind}/{name}" if name else None


def require(row, what: str):
    if row is None:
        raise HTTPException(404, f"{what} not found")
    return row
