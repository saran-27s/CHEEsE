"""Frame-difference occlusion proxy.

This is NOT hand detection. It flags frames where much of the warped board changed since the
previous sampled frame. It misses a hand held still and reacts to lighting changes. Results
that depend on it are labelled degraded.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionOcclusion:
    def __init__(self, fraction_threshold: float = 0.02, pixel_threshold: int = 25):
        self.fraction_threshold = fraction_threshold
        self.pixel_threshold = pixel_threshold
        self._prev: np.ndarray | None = None

    def update(self, warped_bgr: np.ndarray) -> tuple[bool, float]:
        gray = cv2.GaussianBlur(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return False, 0.0
        diff = cv2.absdiff(prev, gray)
        fraction = float(np.count_nonzero(diff > self.pixel_threshold)) / diff.size
        return fraction > self.fraction_threshold, fraction
