"""Video probing and incremental frame sources.

Frames are read one at a time; the video is never loaded into memory. A live source (webcam)
only needs to implement the same ``FrameSource`` iterator protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np
from pydantic import BaseModel

from .images import MediaError

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoMeta(BaseModel):
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int
    timestamp_source: str


@dataclass
class Frame:
    index: int
    t: float  # media seconds
    image: np.ndarray


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...


def check_video_extension(filename: str) -> None:
    ext = Path(filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise MediaError(f"unsupported video extension {ext or '(none)'}; use one of "
                         + ", ".join(sorted(VIDEO_EXTS)), 415)


def probe(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise MediaError("OpenCV could not open the video (unsupported codec or corrupt file)", 422)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise MediaError("video contains no readable frames", 422)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
        if fps <= 0 or fps > 1000:
            raise MediaError(f"video reports an unusable frame rate ({fps}); cannot compute timestamps", 422)
        duration = count / fps if count > 0 else 0.0
        return VideoMeta(fps=fps, frame_count=count, duration=duration, width=w, height=h,
                         timestamp_source="frame_index / container fps (constant frame rate assumed)")
    finally:
        cap.release()


def read_frame_at(path: Path, t: float) -> tuple[np.ndarray, float]:
    """Return (frame, actual_t) closest to media time ``t`` by decoding forward from the start."""
    meta = probe(path)
    target = max(0, min(int(round(t * meta.fps)), max(meta.frame_count - 1, 0)))
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame, target / meta.fps
        # Fallback for containers where seeking fails: decode sequentially.
        cap.release()
        cap = cv2.VideoCapture(str(path))
        last = None
        for i in range(target + 1):
            ok, frame = cap.read()
            if not ok:
                break
            last = (frame, i / meta.fps)
        if last is None:
            raise MediaError("could not read a frame at the requested time", 422)
        return last
    finally:
        cap.release()


class VideoFileFrameSource:
    """Yield frames at roughly ``sample_fps`` using grab() for skipped frames."""

    def __init__(self, path: Path, sample_fps: float, start_t: float = 0.0):
        self.path = path
        self.meta = probe(path)
        self.sample_fps = max(0.1, min(sample_fps, self.meta.fps))
        self.start_t = start_t

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise MediaError("OpenCV could not open the video", 422)
        try:
            fps = self.meta.fps
            interval = 1.0 / self.sample_fps
            next_t = self.start_t
            index = -1
            while True:
                if not cap.grab():
                    break
                index += 1
                t = index / fps
                if t + 1e-9 < next_t:
                    continue
                ok, image = cap.retrieve()
                if not ok or image is None:
                    continue
                next_t += interval
                while next_t <= t:
                    next_t += interval
                yield Frame(index=index, t=t, image=image)
        finally:
            cap.release()
