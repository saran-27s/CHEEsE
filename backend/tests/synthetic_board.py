"""TEST FIXTURES: synthetic rendered boards and a colour-code test detector.

Nothing here is real inference. Pieces are drawn as coloured discs and a "hand" as a magenta
rectangle; ``ColorCodeTestDetector`` reads those colours back. This lets the API and video job
tests exercise calibration, warping, observation building, tracking and persistence end to end
without model weights.
"""

from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from chessvision.core.observation import AnchorSpec, Detection
from chessvision.inference.base import ModelSpec

SIZE = 640
CELL = SIZE // 8
IMG_W, IMG_H = 640, 480
QUAD = [(170.0, 70.0), (480.0, 60.0), (600.0, 440.0), (40.0, 450.0)]  # clockwise, top-left first
A1_INDEX = 3  # a1 at bottom-left: standard view with White nearest the camera

COLORS = {  # BGR, well separated from the grey squares and from each other
    "P": (0, 0, 255), "N": (0, 128, 255), "B": (0, 255, 255), "R": (0, 255, 0), "Q": (255, 0, 0), "K": (255, 255, 0),
    "p": (0, 0, 128), "n": (0, 64, 128), "b": (0, 128, 128), "r": (0, 128, 0), "q": (128, 0, 0), "k": (128, 128, 0),
}
HAND_COLOR = (255, 0, 255)


def render_canonical(board: chess.Board, hand_over: list[str] | None = None) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        row = 7 - r
        shade = 190 if (f + r) % 2 else 70
        img[row * CELL:(row + 1) * CELL, f * CELL:(f + 1) * CELL] = (shade, shade, shade)
        p = board.piece_at(sq)
        if p:
            center = (f * CELL + CELL // 2, row * CELL + CELL // 2)
            cv2.circle(img, center, CELL // 3, COLORS[p.symbol()], -1)
    for name in hand_over or []:
        sq = chess.parse_square(name)
        f, row = chess.square_file(sq), 7 - chess.square_rank(sq)
        cv2.rectangle(img, (f * CELL - 20, row * CELL - 20), ((f + 1) * CELL + 20, (row + 1) * CELL + 20), HAND_COLOR, -1)
    return img


def render_photo(board: chess.Board, hand_over: list[str] | None = None) -> np.ndarray:
    """Project the canonical board into a perspective 'camera' image using QUAD/A1_INDEX."""
    canonical = render_canonical(board, hand_over)
    src = np.float32([(0, SIZE), (0, 0), (SIZE, 0), (SIZE, SIZE)])  # a1, a8, h8, h1
    dst = np.float32([QUAD[(A1_INDEX + j) % 4] for j in range(4)])
    m = cv2.getPerspectiveTransform(src, dst)
    bg = np.full((IMG_H, IMG_W, 3), 30, np.uint8)
    warped = cv2.warpPerspective(canonical, m, (IMG_W, IMG_H), dst=bg, borderMode=cv2.BORDER_TRANSPARENT)
    return warped


def png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


class ColorCodeTestDetector:
    """Reads the coloured discs drawn by this module. Test fixture only."""

    def __init__(self, with_hand: bool = True, fail: bool = False, conf: float = 0.93):
        self.spec = ModelSpec(id="test-colorcode-detector", task="piece_detector", file="(test fixture)",
                              has_hand_class=with_hand, anchor=AnchorSpec(mode="fraction", fx=0.5, fy=0.5),
                              license="test fixture", class_map={})
        self.with_hand = with_hand
        self.fail = fail
        self.conf = conf

    def detect(self, warped: np.ndarray) -> list[Detection]:
        if self.fail:
            raise RuntimeError("simulated detector failure (test fixture)")
        out: list[Detection] = []
        size = warped.shape[0]
        cell = size / 8
        hand = np.all(np.abs(warped.astype(int) - HAND_COLOR) < 60, axis=2)
        if self.with_hand and hand.mean() > 0.01:
            ys, xs = np.nonzero(hand)
            out.append(Detection(label="hand", conf=0.9, box=(float(xs.min()), float(ys.min()),
                                                              float(xs.max()), float(ys.max()))))
        for row in range(8):
            for f in range(8):
                cx, cy = int((f + 0.5) * cell), int((row + 0.5) * cell)
                patch = warped[cy - 6:cy + 6, cx - 6:cx + 6].reshape(-1, 3).mean(axis=0)
                best, dist = None, 1e9
                for label, color in COLORS.items():
                    d = float(np.linalg.norm(patch - np.array(color)))
                    if d < dist:
                        best, dist = label, d
                if dist < 45:
                    out.append(Detection(label=best, conf=self.conf,
                                         box=(f * cell + 10, row * cell + 10, (f + 1) * cell - 10, (row + 1) * cell - 10)))
        return out


def write_video(path: Path, segments: list[tuple[float, chess.Board, list[str] | None]], fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (IMG_W, IMG_H))
    assert writer.isOpened(), "OpenCV VideoWriter (MJPG) unavailable"
    for duration, board, hand in segments:
        frame = render_photo(board, hand)
        for _ in range(int(round(duration * fps))):
            writer.write(frame)
    writer.release()
    return path


def boards(*ucis: str) -> list[chess.Board]:
    b = chess.Board()
    out = [b.copy()]
    for u in ucis:
        b.push_uci(u)
        out.append(b.copy())
    return out
