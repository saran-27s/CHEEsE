"""Application settings.

The scalar fields listed in ``_SCALAR_FIELDS`` can be overridden with an environment variable
named ``CHESSVISION_<FIELD_NAME_UPPER>``, for example ``CHESSVISION_DATA_DIR``. Nested settings
(resolver, observation, tracker) and CORS origins are code defaults only.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

BACKEND_DIR = Path(__file__).resolve().parent.parent


class ResolverSettings(BaseModel):
    uncertain_threshold: float = 0.6
    margin_threshold: float = 3.0
    max_candidates_reported: int = 8
    # Square comparison for video jobs: "colour" (empty / white / black; tolerant of piece-type
    # misreads such as king vs queen), "piece" (exact labels) or "auto" (colour for model input).
    video_match: str = "auto"


class ObservationSettings(BaseModel):
    empty_prior: float = 0.92
    edge_margin: float = 0.12
    hand_dilation: float = 0.10


class TrackerSettings(BaseModel):
    stable_seconds: float = 1.0
    min_frames: int = 3
    dropout_piece_ratio: float = 0.5
    window_tolerance: int = 1  # squares allowed to flicker inside a stable window
    deviation_squares: int = 1  # confident squares that must differ to start settling
    follow_board: bool = True  # re-register the board on every frame (board or camera moved)
    max_plies: int = 3  # longest move sequence searched when one move does not explain the board
    auto_multi_ply: bool = True  # record a unique, clear sequence without asking (fast play)


class Settings(BaseModel):
    data_dir: Path = BACKEND_DIR / "data"
    models_manifest: Path = BACKEND_DIR / "models" / "models.json"
    warp_size: int = 640
    max_image_mb: int = 25
    max_video_mb: int = 2048
    video_sample_fps: float = 5.0
    motion_fraction: float = 0.02
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    resolver: ResolverSettings = Field(default_factory=ResolverSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chessvision.db"

    def dir(self, kind: str) -> Path:
        """Return (and create) a storage subdirectory: uploads, previews, thumbnails, videos."""
        path = self.data_dir / kind
        path.mkdir(parents=True, exist_ok=True)
        return path


_SCALAR_FIELDS = {
    "data_dir": Path,
    "models_manifest": Path,
    "warp_size": int,
    "max_image_mb": int,
    "max_video_mb": int,
    "video_sample_fps": float,
    "motion_fraction": float,
}


def load_settings(**overrides) -> Settings:
    values: dict = {}
    for name, kind in _SCALAR_FIELDS.items():
        raw = os.environ.get(f"CHESSVISION_{name.upper()}")
        if raw is not None:
            values[name] = kind(raw)
    values.update(overrides)
    settings = Settings(**values)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
