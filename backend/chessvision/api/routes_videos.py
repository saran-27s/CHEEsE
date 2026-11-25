from __future__ import annotations

from pathlib import Path
from typing import Literal

import chess
import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..core.calibration import CalibrationError, CalibrationInput, CameraSide, build_calibration, draw_grid
from ..jobs.video_job import capabilities_for
from ..media.images import MediaError, save_jpeg
from ..media.video import check_video_extension, probe, read_frame_at
from ..storage.db import new_id, now_iso
from . import games_service as gs
from .state import AppState, file_url, get_state, require
from .uploads import stream_to_file

router = APIRouter(prefix="/api")


class JobRequest(BaseModel):
    corners: list[tuple[float, float]] = Field(..., min_length=4, max_length=4)
    a1_index: int = Field(..., ge=0, le=3)
    camera_side: CameraSide = "auto"
    calibration_t: float = Field(0.0, ge=0, description="Media time of the frame the corners were clicked on")
    start: gs.StartSpec = Field(default_factory=lambda: gs.StartSpec(kind="start"))
    sample_fps: float | None = Field(None, gt=0, le=60)
    stable_seconds: float | None = Field(None, gt=0, le=30)
    min_frames: int | None = Field(None, ge=1, le=100)
    title: str | None = None


class FramePreviewRequest(BaseModel):
    t: float = Field(..., ge=0)
    corners: list[tuple[float, float]] = Field(..., min_length=4, max_length=4)
    a1_index: int = Field(..., ge=0, le=3)
    camera_side: CameraSide = "auto"


class ReviewRequest(BaseModel):
    event_id: str
    action: Literal["move", "no_move", "stop"]
    uci: str | None = None
    ucis: list[str] | None = Field(None, min_length=1, max_length=8,
                                   description="Several consecutive moves (fast play between two stable views)")


def video_view(row: dict) -> dict:
    return {"id": row["id"], "filename": row["filename"], "meta": row["meta"], "created_at": row["created_at"]}


@router.post("/videos")
async def upload_video(file: UploadFile = File(...), st: AppState = Depends(get_state)):
    check_video_extension(file.filename or "")
    vid = new_id()
    dest = st.settings.dir("videos") / f"{vid}{Path(file.filename or '').suffix.lower()}"
    await stream_to_file(file, dest, st.settings.max_video_mb * 1_000_000)
    try:
        meta = probe(dest)
    except MediaError:
        dest.unlink(missing_ok=True)
        raise
    row = st.db.insert("videos", {"id": vid, "filename": file.filename, "path": str(dest), "meta_json": meta.model_dump()})
    return video_view(row)


@router.get("/videos/{video_id}")
def get_video(video_id: str, st: AppState = Depends(get_state)):
    return video_view(require(st.db.get("videos", video_id), "video"))


@router.get("/videos/{video_id}/frame")
def video_frame(video_id: str, t: float = Query(0.0, ge=0), st: AppState = Depends(get_state)):
    row = require(st.db.get("videos", video_id), "video")
    frame, actual = read_frame_at(Path(row["path"]), t)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(500, "could not encode frame")
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Frame-Time": f"{actual:.3f}"})


@router.post("/videos/{video_id}/calibration-preview")
def video_calibration_preview(video_id: str, req: FramePreviewRequest, st: AppState = Depends(get_state)):
    row = require(st.db.get("videos", video_id), "video")
    frame, actual = read_frame_at(Path(row["path"]), req.t)
    h, w = frame.shape[:2]
    try:
        cal = build_calibration(CalibrationInput(corners=req.corners, a1_index=req.a1_index,
                                                 camera_side=req.camera_side), w, h, st.settings.warp_size)
    except CalibrationError as exc:
        return {"valid": False, "errors": exc.errors, "preview_url": None}
    name = f"v{video_id}_{new_id()[:8]}.jpg"
    save_jpeg(draw_grid(cal.warp(frame)), st.settings.dir("previews") / name)
    return {"valid": True, "errors": [], "preview_url": file_url("previews", name), "frame_time": actual,
            "calibration": cal.to_dict()}


def job_view(st: AppState, job: dict) -> dict:
    events = st.db.query("SELECT * FROM job_events WHERE job_id = ? ORDER BY created_at", (job["id"],))
    pending = None
    for e in events:
        if e["kind"] == "review" and e.get("resolution") is None:
            pending = e["id"]
    return {
        "id": job["id"], "video_id": job["video_id"], "game_id": job["game_id"], "status": job["status"],
        "progress": job["progress"], "media_time": job["media_time"], "tracker_state": job["tracker_state"],
        "params": job["params"], "capabilities": job.get("capabilities"), "error": job["error"],
        "pending_review_event_id": pending if job["status"] == "paused_for_review" else None,
        "events": [{"id": e["id"], "kind": e["kind"], "t_settled": e["t_settled"], "payload": e["payload"],
                    "thumbnail_url": file_url("thumbnails", e["thumbnail"]), "resolution": e.get("resolution")}
                   for e in events],
        "timestamp_note": "t_settled is the media time of the first frame of the accepted stable window "
                          "(board settled), not the moment the piece was touched.",
        "created_at": job["created_at"], "updated_at": job["updated_at"],
    }


@router.post("/videos/{video_id}/jobs")
def start_job(video_id: str, req: JobRequest, st: AppState = Depends(get_state)):
    video = require(st.db.get("videos", video_id), "video")
    meta = video["meta"]
    try:
        build_calibration(CalibrationInput(corners=req.corners, a1_index=req.a1_index, camera_side=req.camera_side),
                          meta["width"], meta["height"], st.settings.warp_size)
    except CalibrationError as exc:
        raise HTTPException(422, {"message": "invalid calibration", "errors": exc.errors})
    piece = st.registry.piece_detector_status()
    if piece.get("status") != "available":
        raise HTTPException(409, "video analysis needs a piece detector: " + piece.get("message", "unavailable"))
    board, start_meta, desc = gs.resolve_start(st.db, req.start)
    ts = now_iso()
    game = st.db.insert("games", {"title": req.title or f"Video {video['filename']}", "start_fen": board.fen(),
                                  "start_metadata_json": start_meta.model_dump(), "source": f"video ({desc})",
                                  "created_at": ts, "updated_at": ts})
    params = {"calibration": {"corners": [list(c) for c in req.corners], "a1_index": req.a1_index,
                              "camera_side": req.camera_side},
              "calibration_t": req.calibration_t,
              "sample_fps": req.sample_fps or st.settings.video_sample_fps,
              "stable_seconds": req.stable_seconds or st.settings.tracker.stable_seconds,
              "min_frames": req.min_frames or st.settings.tracker.min_frames,
              "min_frames_explicit": req.min_frames is not None}
    job = st.db.insert("jobs", {"video_id": video_id, "game_id": game["id"], "status": "queued", "progress": 0.0,
                                "params_json": params,
                                "capabilities_json": capabilities_for(bool(piece.get("has_hand_class")), piece.get("id")),
                                "created_at": ts, "updated_at": ts})
    st.worker.submit(job["id"], st.video_runner)
    return job_view(st, job)


@router.get("/jobs")
def list_jobs(st: AppState = Depends(get_state)):
    rows = st.db.query("SELECT * FROM jobs ORDER BY created_at DESC")
    return [{"id": r["id"], "video_id": r["video_id"], "game_id": r["game_id"], "status": r["status"],
             "progress": r["progress"], "error": r["error"], "created_at": r["created_at"]} for r in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, st: AppState = Depends(get_state)):
    return job_view(st, require(st.db.get("jobs", job_id), "job"))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, st: AppState = Depends(get_state)):
    job = require(st.db.get("jobs", job_id), "job")
    if job["status"] in ("completed", "cancelled", "failed"):
        raise HTTPException(409, f"job already {job['status']}")
    if not st.worker.cancel(job_id):
        raise HTTPException(409, "job is not managed by this backend process")
    return job_view(st, st.db.get("jobs", job_id))


@router.post("/jobs/{job_id}/review")
def review_job(job_id: str, req: ReviewRequest, st: AppState = Depends(get_state)):
    job = require(st.db.get("jobs", job_id), "job")
    if job["status"] != "paused_for_review":
        raise HTTPException(409, f"job is {job['status']}, not paused for review")
    view = job_view(st, job)
    if view["pending_review_event_id"] != req.event_id:
        raise HTTPException(409, "event is not the pending review")
    resolution: dict = {"action": req.action, "resolved_at": now_iso()}
    if req.action == "move":
        ucis = req.ucis or ([req.uci] if req.uci else [])
        if not ucis:
            raise HTTPException(422, "uci or ucis is required for action 'move'")
        board = gs.final_board(st.db, st.db.get("games", job["game_id"]))
        for u in ucis:
            try:
                move = chess.Move.from_uci(u)
            except ValueError:
                raise HTTPException(422, f"malformed UCI {u!r}")
            if move not in board.legal_moves:
                raise HTTPException(422, f"{u} is not legal in {board.fen()}")
            board.push(move)
        resolution["uci"] = ucis[0]
        resolution["ucis"] = ucis
    if not st.worker.resolve_review(job_id, resolution):
        raise HTTPException(409, "job is not managed by this backend process")
    return {"accepted": True, "resolution": resolution}
