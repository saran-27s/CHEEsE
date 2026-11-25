from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import BACKEND_DIR, Settings, load_settings
from ..core.calibration import CalibrationError
from ..inference.base import ModelUnavailable
from ..inference.registry import ModelRegistry
from ..jobs.video_job import VideoJobRunner
from ..jobs.worker import JobWorker
from ..media.images import MediaError
from ..storage.db import Database
from . import routes_games, routes_images, routes_system, routes_videos
from .state import AppState

FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


def create_app(settings: Settings | None = None, registry: ModelRegistry | None = None,
               serve_frontend: bool = True) -> FastAPI:
    settings = settings or load_settings()
    db = Database(settings.db_path)
    registry = registry or ModelRegistry(settings.models_manifest)
    worker = JobWorker(db)
    state = AppState(settings=settings, db=db, registry=registry, worker=worker,
                     video_runner=VideoJobRunner(db, settings, registry))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="chess-vision", version=__version__, lifespan=lifespan)
    app.state.cv = state
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(MediaError)
    async def media_error(_: Request, exc: MediaError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(ModelUnavailable)
    async def model_unavailable(_: Request, exc: ModelUnavailable):
        return JSONResponse({"detail": f"model unavailable: {exc}"}, status_code=409)

    @app.exception_handler(CalibrationError)
    async def calibration_error(_: Request, exc: CalibrationError):
        return JSONResponse({"detail": {"message": "invalid calibration", "errors": exc.errors}}, status_code=422)

    for module in (routes_system, routes_images, routes_games, routes_videos):
        app.include_router(module.router)

    if serve_frontend and Path(FRONTEND_DIST, "index.html").is_file():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app


logging.basicConfig(level=logging.INFO)
