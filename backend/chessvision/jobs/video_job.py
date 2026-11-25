"""Recorded-video processing job: frames -> warp -> detect -> observe -> track -> persist."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess

from ..config import Settings
from ..core.board_motion import BoardMotionTracker
from ..core.calibration import CalibrationError, CalibrationInput, build_calibration
from ..core.notation import describe_move
from ..core.temporal import FrameEvidence, TemporalTracker, TrackerConfig, TrackerEvent
from ..inference.motion import MotionOcclusion
from ..inference.registry import ModelRegistry
from ..media.images import save_jpeg
from ..media.video import VideoFileFrameSource, read_frame_at
from ..pipeline import observe
from ..storage.db import Database, new_id, now_iso
from .worker import JobCancelled, JobControl


def capabilities_for(spec_has_hand: bool, model_id: str | None) -> dict[str, Any]:
    if spec_has_hand:
        return {"model_id": model_id, "hand_detection": True, "occlusion_source": "hand_class", "degraded": False,
                "note": "Occlusion from the model's hand class."}
    return {"model_id": model_id, "hand_detection": False, "occlusion_source": "motion_proxy", "degraded": True,
            "note": "degraded: motion-based occlusion, not hand detection"}


class VideoJobRunner:
    def __init__(self, db: Database, settings: Settings, registry: ModelRegistry):
        self.db = db
        self.settings = settings
        self.registry = registry

    def _thumb(self, image, job_id: str) -> str:
        name = f"{job_id}_{new_id()[:8]}.jpg"
        save_jpeg(image, self.settings.dir("thumbnails") / name, max_side=320)
        return name

    def _progress(self, job_id: str, t: float, duration: float, state: str) -> None:
        progress = min(0.999, t / duration) if duration > 0 else 0.0
        self.db.update("jobs", job_id, {"progress": progress, "media_time": t, "tracker_state": state,
                                        "updated_at": now_iso()})

    def _add_move(self, game_id: str, board_before: chess.Board, uci: str, provenance: str, ev: TrackerEvent,
                  thumb: str | None, evidence=None) -> dict:
        ply = len(self.db.active_moves(game_id)) + 1
        desc = describe_move(board_before, chess.Move.from_uci(uci))
        self.db.insert("moves", {"game_id": game_id, "ply": ply, "uci": uci, "provenance": provenance,
                                 "evidence_json": evidence.model_dump() if evidence is not None else None,
                                 "t_settled": ev.t_settled,
                                 "t_accepted": ev.t_accepted, "thumbnail": thumb, "status": "active"})
        self.db.update("games", game_id, {"updated_at": now_iso()})
        return desc.model_dump()

    def __call__(self, job_id: str, control: JobControl) -> None:
        job = self.db.get("jobs", job_id)
        assert job is not None
        video = self.db.get("videos", job["video_id"])
        assert video is not None
        params = job["params"]
        detector = self.registry.piece_detector()  # raises ModelUnavailable -> job failed with reason
        spec = detector.spec
        caps = capabilities_for(spec.has_hand_class, spec.id)
        self.db.update("jobs", job_id, {"capabilities_json": caps})

        meta = video["meta"]
        cal_in = CalibrationInput(**params["calibration"])
        cal0 = build_calibration(cal_in, meta["width"], meta["height"], self.settings.warp_size)
        cal = cal0
        game = self.db.get("games", job["game_id"])
        assert game is not None
        sample_fps = float(params.get("sample_fps", self.settings.video_sample_fps))
        stable_s = float(params.get("stable_seconds", self.settings.tracker.stable_seconds))
        min_frames = int(params.get("min_frames", self.settings.tracker.min_frames))
        if not params.get("min_frames_explicit"):
            # Frames needed to span stable_s at this sample rate. A fixed count of 3 would demand
            # 2 s of stillness at 1 fps and make results depend on the sample rate (R5.7).
            min_frames = min(min_frames, max(2, int(stable_s * sample_fps + 1e-9) + 1))
        tcfg = TrackerConfig(stable_seconds=stable_s, min_frames=min_frames,
                             glimpse_min_frames=1 if sample_fps < 2 else 2,
                             dropout_piece_ratio=self.settings.tracker.dropout_piece_ratio,
                             uncertain_threshold=self.settings.resolver.uncertain_threshold,
                             margin_threshold=self.settings.resolver.margin_threshold,
                             window_tolerance=self.settings.tracker.window_tolerance,
                             deviation_squares=self.settings.tracker.deviation_squares,
                             match=self.settings.resolver.video_match,
                             max_plies=self.settings.tracker.max_plies,
                             auto_multi_ply=self.settings.tracker.auto_multi_ply)
        tracker = TemporalTracker(chess.Board(game["start_fen"]), tcfg)
        motion = None if spec.has_hand_class else MotionOcclusion(self.settings.motion_fraction)
        source = VideoFileFrameSource(Path(video["path"]), sample_fps)
        duration = meta["duration"]
        last_write = 0.0

        # Board-motion tracking: the calibration frame is the reference for feature registration.
        board_motion: BoardMotionTracker | None = None
        if self.settings.tracker.follow_board:
            ref_t = float(params.get("calibration_t", 0.0) or 0.0)
            ref_frame, _ = read_frame_at(Path(video["path"]), ref_t)
            board_motion = BoardMotionTracker(ref_frame, list(cal_in.corners))
            if not board_motion.usable:
                board_motion = None
        stats = {"frames": 0, "registered": 0, "max_shift_px": 0.0, "hand_frames": 0}
        start_check: dict[str, Any] | None = None
        events_seen = {"move": 0, "review": 0}

        for frame in source:
            if control.cancelled:
                raise JobCancelled()
            stats["frames"] += 1
            if board_motion is not None:
                mr = board_motion.update(frame.image)
                stats["max_shift_px"] = max(stats["max_shift_px"], round(mr.shift_px, 1))
                if mr.registered:
                    stats["registered"] += 1
                    try:
                        cal = replace(build_calibration(
                            CalibrationInput(corners=mr.corners, a1_index=cal_in.a1_index, camera_side="none"),
                            meta["width"], meta["height"], self.settings.warp_size),
                            camera_dir=cal0.camera_dir, camera_side=cal0.camera_side,
                            camera_dir_source=cal0.camera_dir_source)
                    except CalibrationError:
                        pass  # keep the last valid calibration
            warped = cal.warp(frame.image)
            obs = observe(warped, detector, cal, self.settings)
            occluded = obs.hand_detected
            if motion is not None:
                moving, _ = motion.update(warped)
                occluded = occluded or moving
            stats["hand_frames"] += int(occluded)
            if start_check is None and not occluded:
                start_check = self._start_check(tracker, obs, frame.t)
            board_before = tracker.board.copy(stack=False)
            events = tracker.step(FrameEvidence(t=frame.t, observation=obs, occluded=occluded, ref=warped))
            cursor = board_before
            for ev in events:
                if ev.kind in events_seen:
                    events_seen[ev.kind] += 1
                if ev.kind == "move":
                    thumb = self._thumb(ev.ref, job_id) if ev.sequence_last else None
                    # Only the last move of an inferred sequence is explained by the stored evidence on its
                    # own; earlier ones are kept without evidence (revalidated by legality on edits).
                    evidence = ev.evidence if ev.sequence_last else None
                    desc = self._add_move(job["game_id"], cursor, ev.uci, "resolver_video", ev, thumb, evidence)
                    payload = {"uci": ev.uci, "description": desc, "t_accepted": ev.t_accepted,
                               "margin": ev.sequence.margin if ev.sequence else ev.result.margin}
                    if ev.sequence and ev.sequence.best:
                        payload["inferred_sequence"] = {"sans": ev.sequence.best.sans, "index": ev.sequence_index,
                                                        "reason": ev.sequence.reason,
                                                        "orderings": ev.sequence.best.orderings}
                    self.db.insert("job_events", {"job_id": job_id, "kind": "move", "t_settled": ev.t_settled,
                                                  "thumbnail": thumb, "payload_json": payload})
                    cursor = cursor.copy(stack=False)
                    cursor.push_uci(ev.uci)
                elif ev.kind == "review":
                    self._handle_review(job_id, job["game_id"], tracker, ev, control)
            if time.monotonic() - last_write > 0.5:
                self._progress(job_id, frame.t, duration, tracker.state.value)
                last_write = time.monotonic()
        for ev in tracker.finish():
            events_seen["review"] += 1
            self._handle_review(job_id, job["game_id"], tracker, ev, control)
        stats["effective_min_frames"] = min_frames
        self._progress(job_id, duration, duration, tracker.state.value)
        self._summary(job_id, tracker, stats, start_check, events_seen, board_motion is not None, duration)

    def _start_check(self, tracker: TemporalTracker, obs, t: float) -> dict[str, Any]:
        """Compare the first clear frame with the starting position (presence and colour)."""
        th = self.settings.resolver.uncertain_threshold
        wrong: list[str] = []
        for sq in chess.SQUARES:
            s = obs.squares[sq]
            if s.occluded or not s.probs:
                continue
            cls = {"empty": 0.0, "W": 0.0, "B": 0.0}
            for lbl, p in s.probs.items():
                cls["empty" if lbl == "empty" else ("W" if lbl.isupper() else "B")] += p
            top = max(cls, key=cls.get)
            if cls[top] < th:
                continue
            piece = tracker.board.piece_at(sq)
            exp = "empty" if piece is None else ("W" if piece.color else "B")
            if top != exp:
                wrong.append(chess.square_name(sq))
        return {"t": t, "mismatched_squares": wrong}

    def _summary(self, job_id: str, tracker: TemporalTracker, stats: dict, start_check: dict | None,
                 events_seen: dict, follow_board: bool, duration: float) -> None:
        warnings: list[str] = []
        if start_check is None:
            warnings.append("no frame without a hand was found; nothing could be checked")
        elif len(start_check["mismatched_squares"]) > 3:
            warnings.append(f"the first clear frame ({start_check['t']:.2f} s) disagrees with the starting position "
                            f"on {', '.join(start_check['mismatched_squares'])}: check the calibration (a1 corner) "
                            "and the starting FEN")
        if events_seen["move"] == 0 and events_seen["review"] == 0:
            warnings.append("no move was detected in the whole video")
        if tracker.state.value != "STABLE":
            warnings.append(f"the video ended while the tracker was {tracker.state.value}: a change after the last "
                            "recorded move was never confirmed (the board did not stay still long enough at the end)")
        if follow_board and stats["frames"] and stats["registered"] < 0.5 * stats["frames"]:
            warnings.append(f"board tracking succeeded on only {stats['registered']} of {stats['frames']} frames")
        payload = {"moves": events_seen["move"], "reviews": events_seen["review"],
                   "final_state": tracker.state.value, "final_fen": tracker.board.fen(),
                   "follow_board": follow_board, **stats, "start_check": start_check, "warnings": warnings}
        self.db.insert("job_events", {"job_id": job_id, "kind": "summary", "t_settled": duration,
                                      "payload_json": payload})

    def _handle_review(self, job_id: str, game_id: str, tracker: TemporalTracker, ev: TrackerEvent,
                       control: JobControl) -> None:
        thumb = self._thumb(ev.ref, job_id)
        event = self.db.insert("job_events", {
            "job_id": job_id, "kind": "review", "t_settled": ev.t_settled, "thumbnail": thumb,
            "payload_json": {"result": ev.result.model_dump(), "fen": tracker.board.fen(),
                             "sequence": ev.sequence.model_dump() if ev.sequence else None,
                             "end_of_video": ev.end_of_stream,
                             "t_accepted": ev.t_accepted}})
        self.db.update("jobs", job_id, {"status": "paused_for_review", "tracker_state": tracker.state.value,
                                        "media_time": ev.t_accepted, "updated_at": now_iso()})
        resolution = control.wait_for_review()  # raises JobCancelled
        action = resolution.get("action")
        if action == "stop":
            self.db.update("job_events", event["id"], {"resolution_json": resolution})
            control.cancel_event.set()
            raise JobCancelled()
        if action == "move":
            ucis = resolution.get("ucis") or [resolution["uci"]]
            board = tracker.board.copy(stack=False)
            tracker.resolve_review(ucis)
            for i, uci in enumerate(ucis):
                last = i == len(ucis) - 1
                # user-confirmed; the evidence explains only the final position, so attach it to the last move
                self._add_move(game_id, board, uci, "user", ev, thumb if last else None,
                               ev.evidence if last else None)
                board.push_uci(uci)
        else:
            tracker.resolve_review(None)
        self.db.update("job_events", event["id"], {"resolution_json": resolution})
        self.db.update("jobs", job_id, {"status": "running", "tracker_state": tracker.state.value,
                                        "updated_at": now_iso()})
