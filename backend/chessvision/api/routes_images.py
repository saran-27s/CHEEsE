from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import chess
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core.calibration import CalibrationError, CalibrationInput, CameraSide, build_calibration, draw_grid
from ..core.metadata import FIELDS, PositionMetadata, assemble, board_fen_to_placement, default_metadata
from ..core.observation import LABELS, BoardObservation, manual_observation
from ..inference.base import ModelUnavailable
from ..media.images import load_image, save_jpeg, validate_image_bytes
from ..pipeline import observe
from ..storage.db import now_iso
from .routes_system import SINGLE_IMAGE_NOTE
from .state import AppState, file_url, get_state, require
from .uploads import read_limited

router = APIRouter(prefix="/api")


class CalibrationRequest(BaseModel):
    image_id: str
    corners: list[tuple[float, float]] = Field(..., min_length=4, max_length=4)
    a1_index: int = Field(..., ge=0, le=3)
    camera_side: CameraSide = "auto"


class AnalysisRequest(CalibrationRequest):
    mode: Literal["auto", "manual"] = "auto"
    manual_start: Literal["empty", "standard"] = "empty"


class MetadataIn(BaseModel):
    side_to_move: Literal["w", "b"] | None = None
    castling: str | None = None
    en_passant: str | None = None
    halfmove: int | None = Field(None, ge=0)
    fullmove: int | None = Field(None, ge=1)


class CorrectionRequest(BaseModel):
    placement: dict[str, str]  # square -> FEN symbol; omitted squares are empty
    metadata: MetadataIn = Field(default_factory=MetadataIn)


def metadata_from_input(m: MetadataIn) -> PositionMetadata:
    base = default_metadata()
    values = base.model_dump()
    prov = dict(base.provenance)
    for f in FIELDS:
        v = getattr(m, f)
        if v is not None:
            values[f] = v if not isinstance(v, str) else (v.strip() or "-")
            prov[f] = "user"
    values["provenance"] = prov
    return PositionMetadata(**values)


def _calibrate(st: AppState, req: CalibrationRequest):
    img_row = require(st.db.get("images", req.image_id), "image")
    image = load_image(Path(img_row["path"]))
    h, w = image.shape[:2]
    cal = build_calibration(CalibrationInput(corners=req.corners, a1_index=req.a1_index, camera_side=req.camera_side),
                            w, h, st.settings.warp_size)
    return img_row, image, cal


def _save_preview(st: AppState, image_id: str, req: CalibrationRequest, warped) -> str:
    key = hashlib.sha1(repr((req.corners, req.a1_index, req.camera_side)).encode()).hexdigest()[:10]
    name = f"{image_id}_{key}.jpg"
    save_jpeg(draw_grid(warped), st.settings.dir("previews") / name)
    return name


def position_view(analysis: dict) -> dict:
    obs = BoardObservation.model_validate(analysis["observation"])
    corr = analysis.get("correction")
    if corr:
        placement = corr["placement"]
        meta = PositionMetadata.model_validate(corr["metadata"])
        source = "user_correction"
    else:
        placement = obs.placement()
        meta = default_metadata()
        source = "observation"
    check = assemble(placement, meta)
    return {"placement": placement, "placement_source": source, **check.model_dump()}


def analysis_view(row: dict) -> dict:
    return {
        "id": row["id"], "image_id": row["image_id"], "mode": row["mode"], "model": row.get("model"),
        "mode_label": "Automated recognition" if row["mode"] == "model" else "Manual entry (no automated recognition)",
        "calibration": row["calibration"], "preview_url": file_url("previews", row.get("preview")),
        "observation": row["observation"], "correction": row.get("correction"),
        "uncertain_squares": BoardObservation.model_validate(row["observation"]).uncertain_squares(),
        "position": position_view(row), "note": SINGLE_IMAGE_NOTE,
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


@router.post("/images")
async def upload_image(file: UploadFile = File(...), st: AppState = Depends(get_state)):
    max_bytes = st.settings.max_image_mb * 1_000_000
    data = await read_limited(file, max_bytes)
    image = validate_image_bytes(data, file.filename or "", max_bytes)
    ext = Path(file.filename or "").suffix.lower()
    row = st.db.insert("images", {"filename": file.filename, "path": "", "width": image.shape[1],
                                  "height": image.shape[0], "sha256": hashlib.sha256(data).hexdigest()})
    path = st.settings.dir("uploads") / f"{row['id']}{ext}"
    path.write_bytes(data)
    st.db.update("images", row["id"], {"path": str(path)})
    return {"id": row["id"], "filename": file.filename, "width": row["width"], "height": row["height"],
            "url": f"/api/images/{row['id']}/file"}


@router.get("/images/{image_id}/file")
def image_file(image_id: str, st: AppState = Depends(get_state)):
    row = require(st.db.get("images", image_id), "image")
    return FileResponse(row["path"])


@router.post("/images/{image_id}/suggest-corners")
def suggest_corners(image_id: str, st: AppState = Depends(get_state)):
    row = require(st.db.get("images", image_id), "image")
    try:
        detector = st.registry.corner_detector()
    except ModelUnavailable as exc:
        raise HTTPException(409, f"corner suggestion unavailable: {exc.reason}")
    suggestion = detector.suggest(load_image(Path(row["path"])))
    if suggestion is None:
        raise HTTPException(422, "the corner model found no board; click the corners manually")
    return {**suggestion.model_dump(), "note": "Suggestion only. Check every corner and the a1 choice before use."}


@router.post("/calibrations/preview")
def calibration_preview(req: CalibrationRequest, st: AppState = Depends(get_state)):
    try:
        _, image, cal = _calibrate(st, req)
    except CalibrationError as exc:
        return {"valid": False, "errors": exc.errors, "preview_url": None, "calibration": None}
    name = _save_preview(st, req.image_id, req, cal.warp(image))
    return {"valid": True, "errors": [], "preview_url": file_url("previews", name), "calibration": cal.to_dict()}


@router.post("/analyses")
def create_analysis(req: AnalysisRequest, st: AppState = Depends(get_state)):
    try:
        img_row, image, cal = _calibrate(st, req)
    except CalibrationError as exc:
        raise HTTPException(422, {"message": "invalid calibration", "errors": exc.errors})
    warped = cal.warp(image)
    preview = _save_preview(st, req.image_id, req, warped)
    model_info = None
    if req.mode == "auto":
        try:
            detector = st.registry.piece_detector()
        except ModelUnavailable as exc:
            raise HTTPException(409, f"automated recognition unavailable: {exc.reason}. Use manual mode.")
        obs = observe(warped, detector, cal, st.settings)
        mode = "model"
        model_info = {"id": detector.spec.id, "license": detector.spec.license,
                      "anchor": detector.spec.anchor.model_dump(), "has_hand_class": detector.spec.has_hand_class}
    else:
        placement = board_fen_to_placement(chess.STARTING_BOARD_FEN) if req.manual_start == "standard" else {}
        obs = manual_observation(placement)
        mode = "manual"
    ts = now_iso()
    row = st.db.insert("analyses", {
        "image_id": img_row["id"], "calibration_json": cal.to_dict(), "observation_json": obs.model_dump(),
        "mode": mode, "model_json": model_info, "preview": preview, "created_at": ts, "updated_at": ts,
    })
    return analysis_view(row)


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, st: AppState = Depends(get_state)):
    return analysis_view(require(st.db.get("analyses", analysis_id), "analysis"))


@router.put("/analyses/{analysis_id}/correction")
def put_correction(analysis_id: str, req: CorrectionRequest, st: AppState = Depends(get_state)):
    row = require(st.db.get("analyses", analysis_id), "analysis")
    bad = {sq: lbl for sq, lbl in req.placement.items() if lbl not in LABELS or sq not in chess.SQUARE_NAMES}
    if bad:
        raise HTTPException(422, f"invalid squares or labels: {bad}")
    placement = {sq: lbl for sq, lbl in req.placement.items() if lbl != "empty"}
    meta = metadata_from_input(req.metadata)
    obs = BoardObservation.model_validate(row["observation"])
    seen = obs.placement()
    changed = sorted(sq for sq in chess.SQUARE_NAMES if seen.get(sq) != placement.get(sq))
    correction = {"placement": placement, "metadata": meta.model_dump(), "changed_from_observation": changed,
                  "saved_at": now_iso()}
    row = st.db.update("analyses", analysis_id, {"correction_json": correction, "updated_at": now_iso()})
    return analysis_view(row)


@router.get("/analyses/{analysis_id}/report")
def analysis_report(analysis_id: str, st: AppState = Depends(get_state)):
    row = require(st.db.get("analyses", analysis_id), "analysis")
    img = st.db.get("images", row["image_id"])
    view = analysis_view(row)
    return {
        "report_type": "chess-vision image analysis",
        "input": {"image_id": row["image_id"], "filename": img and img["filename"], "sha256": img and img["sha256"],
                  "width": img and img["width"], "height": img and img["height"]},
        "mode": view["mode"], "mode_label": view["mode_label"], "model": view["model"],
        "calibration": view["calibration"],
        "observation": view["observation"],
        "correction": view["correction"],
        "position": view["position"],
        "note": SINGLE_IMAGE_NOTE,
        "generated_at": now_iso(),
    }
