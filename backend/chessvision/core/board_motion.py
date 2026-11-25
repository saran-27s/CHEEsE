"""Keep a video calibration aligned when the board or camera moves.

The user calibrates on one frame. If the board is bumped or the camera shifts afterwards, a
fixed homography maps pieces onto the wrong squares (half a square of drift is enough to make
every later observation look like many simultaneous moves). ``BoardMotionTracker`` matches SIFT
features of the board area between the calibration frame and each later frame, estimates a
homography with RANSAC, and moves the user's four corners with it.

The user's corners and a1 choice stay authoritative: only their image positions are updated.
When a frame cannot be registered (hand covering the board, blur), the last good corners are
kept and the frame is reported as not registered.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionResult:
    corners: list[tuple[float, float]]
    registered: bool  # True when this frame produced a fresh, accepted estimate
    inliers: int
    shift_px: float  # largest corner displacement from the calibration corners
    reason: str = ""


class BoardMotionTracker:
    def __init__(self, reference_bgr: np.ndarray, corners: list[tuple[float, float]], *,
                 max_side: int = 960, min_inliers: int = 25, max_jump_fraction: float = 0.2,
                 ratio: float = 0.75, ransac_px: float = 3.0, mask_grow: float = 1.08,
                 search_grow: float = 1.25):
        self.corners0 = np.asarray(corners, dtype=np.float32)
        self.current = self.corners0.copy()
        h, w = reference_bgr.shape[:2]
        self.scale = min(1.0, max_side / float(max(h, w)))
        self.min_inliers = min_inliers
        self.max_jump_fraction = max_jump_fraction
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.search_grow = search_grow
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = self._gray(reference_bgr)
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(gray, self._mask(gray.shape, self.corners0, mask_grow))
        side = float(np.mean([np.linalg.norm(self.corners0[i] - self.corners0[(i + 1) % 4]) for i in range(4)]))
        self.board_side = side

    @property
    def usable(self) -> bool:
        return self.ref_desc is not None and len(self.ref_kp) >= self.min_inliers

    def _gray(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        if self.scale < 1.0:
            g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return g

    def _mask(self, shape, corners: np.ndarray, grow: float) -> np.ndarray:
        pts = corners * self.scale
        c = pts.mean(axis=0)
        poly = np.round(c + (pts - c) * grow).astype(np.int32)
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(m, poly, 255)
        return m

    def _keep(self, reason: str, inliers: int = 0) -> MotionResult:
        shift = float(np.abs(self.current - self.corners0).max())
        return MotionResult([tuple(map(float, p)) for p in self.current], False, inliers, shift, reason)

    def update(self, frame_bgr: np.ndarray) -> MotionResult:
        if not self.usable:
            return self._keep("reference frame has too few features")
        gray = self._gray(frame_bgr)
        kp, desc = self.sift.detectAndCompute(gray, self._mask(gray.shape, self.current, self.search_grow))
        if desc is None or len(kp) < self.min_inliers:
            return self._keep("too few features in frame")
        pairs = self.matcher.knnMatch(self.ref_desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
        if len(good) < self.min_inliers:
            return self._keep(f"only {len(good)} feature matches")
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < self.min_inliers:
            return self._keep(f"only {n_in} RANSAC inliers", n_in)
        new = cv2.perspectiveTransform((self.corners0 * self.scale).reshape(-1, 1, 2), H).reshape(-1, 2) / self.scale
        if not np.all(np.isfinite(new)):
            return self._keep("degenerate homography", n_in)
        jump = float(np.abs(new - self.current).max())
        if jump > self.max_jump_fraction * self.board_side:
            return self._keep(f"corner jump {jump:.0f}px rejected", n_in)
        self.current = new.astype(np.float32)
        shift = float(np.abs(self.current - self.corners0).max())
        return MotionResult([tuple(map(float, p)) for p in self.current], True, n_in, shift)
