"""Model manifest, availability checks and lazy loading.

The registry never fabricates inference. If a model cannot be used it reports exactly why.
``inject`` exists so automated tests can supply a detector explicitly; injected detectors are
reported in health output with their own id.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

from .base import CornerDetector, ModelSpec, ModelUnavailable, PieceDetector


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.models_dir = manifest_path.parent
        self.specs: list[ModelSpec] = []
        self.manifest_error: str | None = None
        self._loaded: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._hash_cache: dict[str, tuple[float, str]] = {}
        self._injected: dict[str, Any] = {}
        self._lock = threading.Lock()
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.specs = [ModelSpec(**m) for m in data.get("models", [])]
        except FileNotFoundError:
            self.manifest_error = f"manifest not found: {manifest_path}"
        except Exception as exc:  # malformed manifest
            self.manifest_error = f"manifest unreadable: {exc}"

    # ---- inspection ---------------------------------------------------------
    @staticmethod
    def runtime_installed(runtime: str = "ultralytics") -> bool:
        return importlib.util.find_spec(runtime) is not None

    @staticmethod
    def device() -> dict[str, Any]:
        if importlib.util.find_spec("torch") is None:
            return {"torch_installed": False, "device": None, "cuda_available": False}
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            name = torch.cuda.get_device_name(0) if cuda else None
            return {"torch_installed": True, "torch_version": torch.__version__,
                    "device": "cuda:0" if cuda else "cpu", "cuda_available": cuda, "gpu_name": name}
        except Exception as exc:
            return {"torch_installed": True, "device": None, "cuda_available": False, "error": str(exc)}

    def path_of(self, spec: ModelSpec) -> Path:
        p = Path(spec.file)
        return p if p.is_absolute() else self.models_dir / p

    def _file_hash(self, path: Path) -> str:
        mtime = path.stat().st_mtime
        cached = self._hash_cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        digest = _sha256(path)
        self._hash_cache[str(path)] = (mtime, digest)
        return digest

    def spec_status(self, spec: ModelSpec) -> dict[str, Any]:
        path = self.path_of(spec)
        base = {"id": spec.id, "task": spec.task, "path": str(path), "license": spec.license,
                "source_url": spec.source_url, "has_hand_class": spec.has_hand_class,
                "anchor": spec.anchor.model_dump(), "injected": False}
        if spec.id in self._errors:
            return {**base, "status": "load_error", "message": self._errors[spec.id]}
        if not path.exists():
            return {**base, "status": "missing_file", "message": f"weights not found at {path}"}
        if spec.sha256:
            digest = self._file_hash(path)
            if digest.lower() != spec.sha256.lower():
                return {**base, "status": "checksum_mismatch",
                        "message": f"sha256 {digest} does not match manifest {spec.sha256}"}
        if not self.runtime_installed(spec.runtime):
            return {**base, "status": "runtime_not_installed",
                    "message": f"optional runtime '{spec.runtime}' is not installed "
                               "(pip install -r requirements-inference.txt; AGPL-3.0)"}
        return {**base, "status": "available", "message": "ready" if spec.id in self._loaded else "ready (loads on first use)"}

    def status(self) -> list[dict[str, Any]]:
        out = [self.spec_status(s) for s in self.specs]
        for task, det in self._injected.items():
            out.append({"id": det.spec.id, "task": task, "status": "available", "injected": True,
                        "message": "explicitly injected detector (automated tests only)",
                        "has_hand_class": det.spec.has_hand_class, "license": det.spec.license,
                        "path": None, "source_url": None, "anchor": det.spec.anchor.model_dump()})
        return out

    # ---- access -------------------------------------------------------------
    def inject(self, task: str, detector: Any) -> None:
        self._injected[task] = detector

    def _first_spec(self, task: str) -> ModelSpec | None:
        return next((s for s in self.specs if s.task == task), None)

    def _load(self, spec: ModelSpec) -> Any:
        with self._lock:
            if spec.id in self._loaded:
                return self._loaded[spec.id]
            st = self.spec_status(spec)
            if st["status"] != "available":
                raise ModelUnavailable(spec.id, st["message"])
            try:
                from .yolo import UltralyticsCornerDetector, UltralyticsPieceDetector

                cls = UltralyticsPieceDetector if spec.task == "piece_detector" else UltralyticsCornerDetector
                det = cls(spec, self.path_of(spec))
            except Exception as exc:
                self._errors[spec.id] = f"failed to load: {exc}"
                raise ModelUnavailable(spec.id, self._errors[spec.id]) from exc
            self._loaded[spec.id] = det
            return det

    def piece_detector(self) -> PieceDetector:
        if "piece_detector" in self._injected:
            return self._injected["piece_detector"]
        spec = self._first_spec("piece_detector")
        if spec is None:
            raise ModelUnavailable("piece_detector", self.manifest_error or "no piece_detector in manifest")
        return self._load(spec)

    def corner_detector(self) -> CornerDetector:
        if "board_corners" in self._injected:
            return self._injected["board_corners"]
        spec = self._first_spec("board_corners")
        if spec is None:
            raise ModelUnavailable("board_corners", self.manifest_error or "no board_corners model in manifest")
        return self._load(spec)

    def piece_detector_status(self) -> dict[str, Any]:
        if "piece_detector" in self._injected:
            return next(s for s in self.status() if s["task"] == "piece_detector" and s["injected"])
        spec = self._first_spec("piece_detector")
        if spec is None:
            return {"id": None, "status": "missing_file",
                    "message": self.manifest_error or "no piece_detector in manifest", "has_hand_class": False}
        return self.spec_status(spec)
