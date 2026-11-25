"""Optional Ultralytics (AGPL-3.0) adapter. Imported only when the runtime is installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.observation import Detection
from .base import CornerSuggestion, ModelSpec


def _device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing or broken
        return "cpu"


class UltralyticsPieceDetector:
    def __init__(self, spec: ModelSpec, path: Path):
        from ultralytics import YOLO  # noqa: PLC0415 - optional dependency

        self.spec = spec
        self.device = _device()
        self.model = YOLO(str(path))
        self.names: dict[int, str] = dict(self.model.names)
        unknown = [n for n in self.names.values() if n not in spec.class_map]
        if unknown:
            raise ValueError(f"model classes not in manifest class_map: {unknown}")

    def detect(self, warped_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(warped_bgr, imgsz=self.spec.imgsz, conf=self.spec.conf,
                                     device=self.device, verbose=False)
        out: list[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = self.spec.class_map[self.names[int(k)]]
            out.append(Detection(label=label, conf=float(c), box=(float(x1), float(y1), float(x2), float(y2))))
        return out


class UltralyticsCornerDetector:
    def __init__(self, spec: ModelSpec, path: Path):
        from ultralytics import YOLO  # noqa: PLC0415

        self.spec = spec
        self.device = _device()
        self.model = YOLO(str(path))
        if sorted(spec.keypoint_order) != ["a1", "a8", "h1", "h8"]:
            raise ValueError("board_corners model needs keypoint_order with a1, h1, a8, h8")

    def suggest(self, image_bgr: np.ndarray) -> CornerSuggestion | None:
        results = self.model.predict(image_bgr, imgsz=self.spec.imgsz, conf=self.spec.conf,
                                     device=self.device, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None
        kp = results[0].keypoints
        xy = kp.xy[0].cpu().numpy()
        kconf = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(len(xy))
        named = {name: (float(xy[i][0]), float(xy[i][1])) for i, name in enumerate(self.spec.keypoint_order)}
        ring = [named["a1"], named["a8"], named["h8"], named["h1"]]  # clockwise from a1
        start = min(range(4), key=lambda i: ring[i][0] + ring[i][1])  # top-left-most first
        corners = ring[start:] + ring[:start]
        a1_index = (4 - start) % 4
        return CornerSuggestion(corners=corners, a1_index=a1_index, confidence=float(np.mean(kconf)),
                                model_id=self.spec.id)
