"""Manual board calibration: four clicked corners + a1 identification -> homography.

The user clicks the four board corners clockwise on screen (the UI asks to start at the
top-left as seen in the image) and says which clicked point is the a1 corner. Clockwise
from a1, seen from above, the board corners are a1, a8, h8, h1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Chess corners in clockwise order starting at a1, and their canonical warped positions
# (as fractions of the warp size).
CHESS_CORNERS = ("a1", "a8", "h8", "h1")
CANONICAL_UNIT = ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


CameraSide = Literal["auto", "none", "rank1", "rank8", "afile", "hfile"]

# Unit direction from the board centre toward each board side in the canonical warp (y down).
_SIDE_DIRS = {"rank1": (0.0, 1.0), "rank8": (0.0, -1.0), "afile": (-1.0, 0.0), "hfile": (1.0, 0.0)}


class CalibrationInput(BaseModel):
    corners: list[tuple[float, float]] = Field(..., description="Four image points, clockwise on screen")
    a1_index: int = Field(..., ge=0, le=3, description="Index into corners of the a1 corner")
    camera_side: CameraSide = Field(
        "auto", description="Board side nearest the camera. 'auto' estimates it from perspective; "
                            "'none' = top-down or already-warped image (anchor at box centre).")


class CalibrationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _shoelace(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0


def validate_corners(corners: list[tuple[float, float]], width: int, height: int) -> list[str]:
    """Return every geometric problem found (empty list = valid)."""
    errors: list[str] = []
    if len(corners) != 4:
        return [f"expected 4 corners, got {len(corners)}"]
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return ["corners must be four finite (x, y) points"]

    mx, my = 0.05 * width, 0.05 * height
    for i, (x, y) in enumerate(pts):
        if x < -mx or y < -my or x > width + mx or y > height + my:
            errors.append(f"corner {i + 1} ({x:.0f}, {y:.0f}) is outside the image")

    area = _shoelace(pts)  # positive = clockwise on screen (y down)
    if area < 0:
        errors.append("corners are counter-clockwise; click them clockwise on screen")
    if abs(area) < 0.01 * width * height:
        errors.append("board area is below 1% of the image; corners too close together")

    diag = math.hypot(width, height)
    for i in range(4):
        if np.linalg.norm(pts[(i + 1) % 4] - pts[i]) < 0.01 * diag:
            errors.append(f"edge {i + 1}-{(i + 1) % 4 + 1} is too short")

    crosses = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        ab, bc = b - a, c - b
        crosses.append(ab[0] * bc[1] - ab[1] * bc[0])
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        errors.append("corners do not form a convex quadrilateral (self-intersecting or concave)")
    return errors


@dataclass(frozen=True)
class Calibration:
    """A validated mapping from image pixels to the canonical warped board."""

    corners: tuple[tuple[float, float], ...]
    a1_index: int
    size: int
    matrix: np.ndarray  # 3x3 image -> warped
    camera_dir: tuple[float, float] | None
    image_size: tuple[int, int]
    camera_side: str = "auto"
    camera_dir_source: str = "unknown"

    @property
    def chess_corners(self) -> dict[str, tuple[float, float]]:
        return {name: self.corners[(self.a1_index + j) % 4] for j, name in enumerate(CHESS_CORNERS)}

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.matrix, (self.size, self.size), flags=cv2.INTER_LINEAR)

    def image_to_board(self, x: float, y: float) -> tuple[float, float] | None:
        """Project an image point onto the board plane; None if it lies beyond the horizon."""
        v = self.matrix @ np.array([x, y, 1.0])
        ref = self.matrix @ np.array([*self.corners[0], 1.0])
        if abs(v[2]) < 1e-12 or np.sign(v[2]) != np.sign(ref[2]):
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_dict(self) -> dict:
        return {
            "corners": [list(c) for c in self.corners],
            "a1_index": self.a1_index,
            "size": self.size,
            "chess_corners": {k: list(v) for k, v in self.chess_corners.items()},
            "camera_dir": list(self.camera_dir) if self.camera_dir else None,
            "camera_side": self.camera_side,
            "camera_dir_source": self.camera_dir_source,
            "image_size": list(self.image_size),
        }


def build_calibration(inp: CalibrationInput, width: int, height: int, size: int = 640) -> Calibration:
    errors = validate_corners(list(inp.corners), width, height)
    if errors:
        raise CalibrationError(errors)
    src = np.array([inp.corners[(inp.a1_index + j) % 4] for j in range(4)], dtype=np.float32)
    dst = np.array([(u * size, v * size) for u, v in CANONICAL_UNIT], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    corners = tuple((float(x), float(y)) for x, y in inp.corners)
    if inp.camera_side == "auto":
        camera_dir = estimate_camera_dir(corners, inp.a1_index)
        source = "estimated_from_perspective" if camera_dir else "none_detected (near top-down)"
    elif inp.camera_side == "none":
        camera_dir, source = None, "user: top-down / pre-warped"
    else:
        camera_dir, source = _SIDE_DIRS[inp.camera_side], f"user: {inp.camera_side}"
    return Calibration(corners=corners, a1_index=inp.a1_index, size=size, matrix=matrix, camera_dir=camera_dir,
                       image_size=(width, height), camera_side=inp.camera_side, camera_dir_source=source)


def estimate_camera_dir(corners, a1_index: int, min_strength: float = 0.04) -> tuple[float, float] | None:
    """Direction (canonical warped space) from the board centre toward the camera.

    Uses perspective foreshortening: the board edge nearest the camera appears longest in the
    image. Opposite edges of equal length (top-down or already-warped images) give None.
    This is a heuristic; the user can override it with ``camera_side``.
    """
    ring = [np.asarray(corners[(a1_index + j) % 4], dtype=np.float64) for j in range(4)]  # a1, a8, h8, h1
    a1, a8, h8, h1 = ring
    rank1, rank8 = np.linalg.norm(h1 - a1), np.linalg.norm(h8 - a8)
    file_a, file_h = np.linalg.norm(a8 - a1), np.linalg.norm(h8 - h1)
    dy = (rank1 - rank8) / (rank1 + rank8)  # >0: rank 1 nearer -> camera toward +y (bottom)
    dx = (file_h - file_a) / (file_h + file_a)  # >0: h-file nearer -> camera toward +x (right)
    norm = math.hypot(dx, dy)
    if norm < min_strength:
        return None
    return dx / norm, dy / norm


def draw_grid(warped: np.ndarray) -> np.ndarray:
    """Return a copy of the warped board with square boundaries and labels for previews."""
    out = warped.copy()
    size = out.shape[0]
    cell = size / 8.0
    for i in range(9):
        p = int(round(i * cell))
        cv2.line(out, (p, 0), (p, size - 1), (0, 200, 255), 1)
        cv2.line(out, (0, p), (size - 1, p), (0, 200, 255), 1)
    for file in range(8):
        for row in range(8):
            name = f"{'abcdefgh'[file]}{8 - row}"
            org = (int(file * cell + 3), int(row * cell + 14))
            cv2.putText(out, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out
