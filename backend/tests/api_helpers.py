from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient

from chessvision.api.app import create_app
from chessvision.config import BACKEND_DIR, load_settings


def make_client(tmp_path: Path, **overrides) -> TestClient:
    """App with an isolated data dir and a copy of the manifest whose weights are absent."""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy(BACKEND_DIR / "models" / "models.json", models / "models.json")
    settings = load_settings(data_dir=tmp_path / "data", models_manifest=models / "models.json", **overrides)
    return TestClient(create_app(settings, serve_frontend=False))


def wait_job(client: TestClient, job_id: str, statuses: set[str], timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job did not reach {statuses}; last={job['status']} error={job.get('error')}")
