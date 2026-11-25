from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .. import __version__
from ..jobs.video_job import capabilities_for
from .state import SAFE_NAME, AppState, get_state

router = APIRouter(prefix="/api")

SINGLE_IMAGE_NOTE = ("A single image provides piece-placement evidence only. It does not show the last move, "
                     "game history, side to move, castling rights, en passant availability or move counters.")


@router.get("/health")
def health(st: AppState = Depends(get_state)):
    piece = st.registry.piece_detector_status()
    automated = piece.get("status") == "available"
    return {
        "status": "ok",
        "version": __version__,
        "mode": "automated" if automated else "manual",
        "mode_label": "Automated recognition available" if automated
        else "Manual mode (no automated recognition): " + piece.get("message", "piece model unavailable"),
        "piece_detector": piece,
        "occlusion": capabilities_for(bool(piece.get("has_hand_class")), piece.get("id")) if automated else None,
        "models": st.registry.status(),
        "manifest_error": st.registry.manifest_error,
        "runtime_installed": {"ultralytics": st.registry.runtime_installed("ultralytics")},
        "device": st.registry.device(),
        "worker": st.worker.info(),
        "limits": {"max_image_mb": st.settings.max_image_mb, "max_video_mb": st.settings.max_video_mb},
        "settings": {
            "warp_size": st.settings.warp_size,
            "video_sample_fps": st.settings.video_sample_fps,
            "resolver": st.settings.resolver.model_dump(),
            "tracker": st.settings.tracker.model_dump(),
            "observation": st.settings.observation.model_dump(),
            "motion_fraction": st.settings.motion_fraction,
            "data_dir": str(st.settings.data_dir),
        },
        "notes": {"single_image": SINGLE_IMAGE_NOTE},
    }


@router.get("/files/{kind}/{name}")
def get_file(kind: str, name: str, st: AppState = Depends(get_state)):
    if kind not in ("previews", "thumbnails") or not SAFE_NAME.match(name):
        raise HTTPException(404, "file not found")
    path = st.settings.dir(kind) / name
    if not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path)
