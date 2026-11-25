"""One background worker thread with a FIFO queue, progress, cancellation and review pauses."""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from typing import Any, Callable

from ..storage.db import Database, now_iso

log = logging.getLogger(__name__)

TERMINAL = {"completed", "cancelled", "failed"}


class JobCancelled(Exception):
    pass


class JobControl:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.cancel_event = threading.Event()
        self.resume_event = threading.Event()
        self.resolution: dict[str, Any] | None = None

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def wait_for_review(self, poll: float = 0.25) -> dict[str, Any]:
        """Block until the user resolves the pending review. Raises JobCancelled on cancel."""
        self.resume_event.clear()
        self.resolution = None
        while not self.resume_event.wait(poll):
            if self.cancelled:
                raise JobCancelled()
        if self.cancelled:
            raise JobCancelled()
        assert self.resolution is not None
        return self.resolution


Runner = Callable[[str, JobControl], None]


class JobWorker:
    def __init__(self, db: Database):
        self.db = db
        self._queue: "queue.Queue[tuple[str, Runner] | None]" = queue.Queue()
        self._controls: dict[str, JobControl] = {}
        self._current: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        # Jobs that were active when the process stopped cannot be resumed.
        for row in self.db.query("SELECT id FROM jobs WHERE status IN ('queued','running','paused_for_review')"):
            self.db.update("jobs", row["id"], {"status": "failed", "error": "interrupted by backend restart",
                                               "updated_at": now_iso()})
        self._thread = threading.Thread(target=self._loop, name="chessvision-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        for control in self._controls.values():
            control.cancel_event.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def info(self) -> dict[str, Any]:
        return {"alive": self.alive, "current_job": self._current, "queued": self._queue.qsize()}

    def submit(self, job_id: str, runner: Runner) -> None:
        with self._lock:
            self._controls[job_id] = JobControl(job_id)
        self._queue.put((job_id, runner))

    def cancel(self, job_id: str) -> bool:
        control = self._controls.get(job_id)
        if control is None:
            return False
        control.cancel_event.set()
        job = self.db.get("jobs", job_id)
        if job and job["status"] == "queued":
            self.db.update("jobs", job_id, {"status": "cancelled", "updated_at": now_iso()})
        return True

    def resolve_review(self, job_id: str, resolution: dict[str, Any]) -> bool:
        control = self._controls.get(job_id)
        if control is None:
            return False
        control.resolution = resolution
        control.resume_event.set()
        return True

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job_id, runner = item
            control = self._controls[job_id]
            job = self.db.get("jobs", job_id)
            if job is None or control.cancelled or job["status"] in TERMINAL:
                continue
            self._current = job_id
            self.db.update("jobs", job_id, {"status": "running", "updated_at": now_iso()})
            try:
                runner(job_id, control)
                final = "cancelled" if control.cancelled else "completed"
                values: dict[str, Any] = {"status": final, "updated_at": now_iso()}
                if final == "completed":
                    values["progress"] = 1.0
                self.db.update("jobs", job_id, values)
            except JobCancelled:
                self.db.update("jobs", job_id, {"status": "cancelled", "updated_at": now_iso()})
            except Exception as exc:  # report failures instead of losing them
                log.error("job %s failed: %s", job_id, traceback.format_exc())
                self.db.update("jobs", job_id, {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                                "updated_at": now_iso()})
            finally:
                self._current = None
