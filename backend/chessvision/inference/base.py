"""Small interfaces between the pipeline and any model runtime."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from ..core.observation import AnchorSpec, Detection


class ModelSpec(BaseModel):
    id: str
    task: Literal["piece_detector", "board_corners"]
    file: str
    runtime: Literal["ultralytics"] = "ultralytics"
    input: Literal["warped_board", "raw_image"] = "warped_board"
    imgsz: int = 640
    conf: float = 0.25
    class_map: dict[str, str] = Field(default_factory=dict)  # model class name -> our label
    has_hand_class: bool = False
    keypoint_order: list[str] = Field(default_factory=list)  # for board_corners models
    anchor: AnchorSpec = Field(default_factory=AnchorSpec)
    license: str = "unknown"
    source_url: str | None = None
    sha256: str | None = None
    notes: str | None = None


class ModelUnavailable(RuntimeError):
    def __init__(self, model_id: str, reason: str):
        super().__init__(f"{model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


class CornerSuggestion(BaseModel):
    corners: list[tuple[float, float]]  # clockwise on screen, first = top-left-most
    a1_index: int
    confidence: float
    model_id: str


@runtime_checkable
class PieceDetector(Protocol):
    spec: ModelSpec

    def detect(self, warped_bgr: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class CornerDetector(Protocol):
    spec: ModelSpec

    def suggest(self, image_bgr: np.ndarray) -> CornerSuggestion | None: ...
