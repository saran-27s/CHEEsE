"""Shared per-image step: warped board + detector -> observation."""

from __future__ import annotations

import numpy as np

from .config import Settings
from .core.calibration import Calibration
from .core.observation import BoardObservation, build_observation
from .inference.base import PieceDetector


def observe(warped: np.ndarray, detector: PieceDetector, cal: Calibration, settings: Settings) -> BoardObservation:
    detections = detector.detect(warped)
    o = settings.observation
    return build_observation(
        detections, cal.size, anchor=detector.spec.anchor, camera_dir=cal.camera_dir,
        empty_prior=o.empty_prior, uncertain_threshold=settings.resolver.uncertain_threshold,
        edge_margin=o.edge_margin, hand_dilation=o.hand_dilation, source="model",
    )
