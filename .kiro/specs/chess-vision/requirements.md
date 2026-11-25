# Requirements: chess-vision (release 1)

Local-first web app for a solo developer. Inputs: still images and recorded video files. Webcam/live input is out of scope for this release, but the processing pipeline must accept any frame source.

Terminology used throughout:

- **Observation**: per-square evidence produced from one image or frame (model output or manual entry). Never treated as game state.
- **Confirmed position / game**: positions and moves the system or the user has committed. Only the resolver or an explicit user action changes them.
- **Correction**: a user edit to an observation or to a confirmed move. Stored with provenance.
- **Position metadata**: side to move, castling rights, en passant square, halfmove clock, fullmove number.

## R1 Health and settings

- R1.1 WHEN the UI loads THEN it SHALL show backend reachability, API version, model availability per model, inference device, and the worker/job status.
- R1.2 WHEN a configured model file is missing or its optional runtime (`ultralytics`) is not installed THEN health SHALL report the exact reason and the expected path, and the image workflow SHALL be labelled "Manual mode (no automated recognition)".
- R1.3 The backend SHALL never substitute synthetic or fake inference at runtime.
- R1.4 WHEN the piece model has no hand class or hand detection is unavailable THEN health and video results SHALL state "degraded: motion-based occlusion, not hand detection".

## R2 Coordinates, calibration, orientation

- R2.1 The system SHALL define one canonical board frame: warped square image, `a8` top-left, `h1` bottom-right, file index 0..7 = a..h, rank index 0..7 = 1..8.
- R2.2 The user SHALL supply four corner points clicked clockwise starting with the top-left corner as it appears in the image, plus which of the four is the a1 corner.
- R2.3 WHEN corners are submitted THEN the backend SHALL reject them with a specific message if they are non-finite, out of the image by more than 5%, not clockwise, not convex, self-intersecting, or the quadrilateral area is below 1% of the image area.
- R2.4 WHEN calibration is valid THEN the backend SHALL return a corrected (warped) board preview with square labels.
- R2.5 Square mapping SHALL be tested for all four a1 positions.
- R2.6 Piece anchoring SHALL be configurable per model (`fraction` or `toward_camera`) and documented as approximate for oblique cameras and tall pieces.

## R3 Analyze image

- R3.1 The user SHALL be able to upload a JPEG/PNG/WebP (max size configurable, default 25 MB) and preview it. Invalid type, undecodable data, or oversize SHALL return HTTP 4xx with a reason.
- R3.2 WHEN a piece model is available THEN analysis SHALL return per-square label probabilities, the top label, uncertainty flags, and raw detections.
- R3.3 Squares SHALL be flagged uncertain when the top probability is below the configured threshold, when more than one detection maps to the square, or when the anchor lies within the configured margin of a square edge.
- R3.4 The UI SHALL show a digital board with uncertain squares highlighted and allow any square to be set to empty or any piece.
- R3.5 WHEN no model is available THEN the same board editor SHALL be shown, starting empty or from the start position, labelled manual.
- R3.6 A single image SHALL be described as piece-placement evidence only. The response SHALL NOT claim a last move or game history.
- R3.7 FEN export SHALL include metadata provenance per field (`user`, `trusted_fen`, `standard_start`, `derived_from_game`, `assumed_default`). Defaults SHALL be visibly labelled as assumptions. Castling rights SHALL never be inferred from piece placement.

## R4 Compare positions

- R4.1 The "before" position SHALL come from one of: a confirmed analysis of a before-image (with user metadata), a user-supplied FEN, or the standard starting position.
- R4.2 The "after" evidence SHALL come from an analysis of an after-image (model or manual), calibrated independently of the before-image.
- R4.3 The result SHALL be exactly one of `no_move`, `confirmed`, `ambiguous`, `invalid`, with the ranked candidate list and per-candidate scores.
- R4.4 The resolver SHALL NOT return `confirmed` when (a) the best candidate's margin over the second is below the configured threshold, (b) any square changed by the best candidate is occluded or unknown, or (c) the best candidate disagrees with any confidently observed square.
- R4.5 WHEN the observation is not explained by any single legal move or by no move THEN the result SHALL be `invalid` (possible multiple moves or recognition error), not a forced move.
- R4.6 Candidates SHALL include no-move and every legal move, with each promotion choice as a separate candidate; captures, castling and en passant SHALL be scored by their full square changes.
- R4.7 For `confirmed`, the output SHALL include SAN, UCI, readable coordinates (`g1 → f3`), deterministic English, and capture / castling / promotion / check / checkmate / en passant flags.

## R5 Analyze recorded video

- R5.1 The user SHALL upload a video (mp4/mov/avi/mkv/webm, size limit configurable, default 2 GB). The backend SHALL validate that OpenCV can open it and read at least one frame, and return duration, fps, frame count and resolution.
- R5.2 The user SHALL choose a calibration frame by timestamp, calibrate as in R2, and confirm the starting position (standard by default, or a valid custom FEN).
- R5.3 Processing SHALL run in one background worker, read frames incrementally at a configurable sample rate, and never load the whole video into memory.
- R5.4 The job SHALL expose status (`queued`, `running`, `paused_for_review`, `completed`, `cancelled`, `failed`), progress 0..1, current media time, and an error message on failure.
- R5.5 The user SHALL be able to cancel a queued, running or paused job; the worker SHALL stop within one processed frame.
- R5.6 The temporal state machine SHALL use states `STABLE`, `OCCLUDED_OR_MOVING`, `SETTLING`, `VALIDATING`, `NEEDS_REVIEW`.
- R5.7 Stability SHALL be measured in media seconds (configurable, default 1.0 s) and a minimum frame count (default 3), so results do not depend on frame rate.
- R5.8 During occlusion the last confirmed position SHALL be preserved. A piece hidden or missed SHALL NOT be treated as a capture.
- R5.9 A move SHALL be committed only via the resolver rules of R4.4. Each committed move SHALL record the media timestamp of the first frame of the accepted stable window and a thumbnail. The UI SHALL describe this time as "board settled at", not the time the piece was touched.
- R5.10 WHEN the resolver returns `ambiguous` or `invalid` THEN the job SHALL pause with status `paused_for_review`, store a review event with candidates and thumbnail, and continue only after the user picks a candidate, enters a move, declares no move, or stops the job.
- R5.11 The system SHALL NOT claim automatic recovery after missed moves.

## R6 Results, correction, recovery

- R6.1 Games SHALL store the start position with metadata provenance and an ordered list of moves with provenance (`resolver_image`, `resolver_video`, `user`), evidence and timestamps.
- R6.2 WHEN the user edits or deletes move N THEN moves after N SHALL be replayed: moves with stored evidence are re-resolved from the new position and kept only if the resolver confirms the same move; moves without evidence are kept only if legal. The first failing move and all later moves SHALL be removed from the game and returned as `detached` for review.
- R6.3 Observations, corrections and confirmed moves SHALL be stored separately and remain distinguishable in the API and UI.

## R7 Exports

- R7.1 PGN for the confirmed game, including `SetUp`/`FEN` headers for non-standard starts.
- R7.2 FEN for any confirmed position, with a JSON metadata block showing provenance.
- R7.3 JSON analysis report: input references, calibration, observations, corrections, confirmed moves with all notation fields, assumptions, and model information.
- R7.4 Video move events SHALL include timestamps and thumbnail URLs.

## R8 Non-functional

- R8.1 Runs locally: FastAPI + SQLite + local files; React/TypeScript/Vite UI. No paid APIs, hosted LLMs, cloud services, auth, Redis, Celery or microservices.
- R8.2 Notation and English text SHALL be generated from python-chess moves, not from a language model.
- R8.3 Third-party licenses SHALL be recorded in `THIRD_PARTY_NOTICES.md`; model weights and datasets SHALL be checked separately from code.
- R8.4 Unit tests SHALL use synthetic observations clearly named as fixtures. Real-model tests SHALL be skipped, with a reason, when weights are absent.
