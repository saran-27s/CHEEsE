# Tasks: chess-vision (release 1)

Status key: `[ ]` todo, `[x]` done, `[~]` done with a documented gap, `[!]` blocked.

## 0. Audit
- [x] 0.1 Clone reference into a temporary directory, record SHA, read source and notebook. (audit)
- [x] 0.2 Check code, weight and dataset licenses separately. (R8.3)
- [x] 0.3 Write `docs/reference-audit.md`.

## 1. Backend foundation
- [x] 1.1 Create `backend/` package, `requirements*.txt`, virtualenv, pytest config. (R8.1)
- [x] 1.2 `config.py` with env overrides and data directory creation.
- [x] 1.3 SQLite schema and repository helpers. (R6.3, R12)

## 2. Core domain
- [x] 2.1 `coords.py`: canonical frame, pixel/square mapping without clamping. (R2.1)
- [x] 2.2 `calibration.py`: validation errors, homography, warp, camera direction. (R2.2-R2.4)
  - Camera direction was redesigned after the real-model test failed: foreshortening estimate plus a user `camera_side` override (design.md section 3).
- [x] 2.3 Tests: every a1 orientation maps correctly; each validation rule rejects. (R2.5)
- [x] 2.4 `observation.py`: anchors, square distributions, uncertainty, occlusion, manual observations. (R2.6, R3.2, R3.3)
- [x] 2.5 `metadata.py`: provenance, defaults as assumptions, castling validation without inference. (R3.7)
- [x] 2.6 `notation.py`: SAN, UCI, coordinates, English, flags. (R4.7, R8.2)
- [x] 2.7 `resolver.py`: candidates incl. no-move and all promotions; decision rules. (R4.3-R4.6)
- [x] 2.8 `temporal.py`: state machine, time-based stability, dropouts, review handling. (R5.6-R5.11)
- [x] 2.9 `game.py`: replay with evidence re-resolution, detached moves, PGN. (R6.1-R6.2, R7.1)
- [x] 2.10 Tests for 2.4-2.9 using named synthetic fixtures. (R8.4)

## 3. Inference and media
- [x] 3.1 Model manifest + registry with availability reasons, lazy load, device report. (R1.1-R1.2)
- [x] 3.2 Optional Ultralytics adapter (pieces + corners) behind protocols. (R3.2)
- [~] 3.3 Motion occlusion proxy, degraded labelling. (R1.4)
  - The reference piece model has a `Hand` class, so with it occlusion comes from hand boxes and the motion proxy is not used. The proxy (labelled degraded) applies only to models without a hand class; it misses a still hand and reacts to lighting changes. A hand or arm the model fails to box is not treated as occlusion.
- [x] 3.4 Image validation and decoding. (R3.1)
- [~] 3.5 Video probe, streaming upload, `VideoFileFrameSource`. (R5.1, R5.3)
  - Timestamps assume constant frame rate (frame index / fps); variable-frame-rate files may drift. Reported as `timestamp_source`.

## 4. Jobs and API
- [~] 4.1 Single worker with queue, progress, cancel, review pause, restart recovery. (R5.3-R5.5, R5.10)
  - Restart recovery marks interrupted jobs `failed` ("interrupted by backend restart"); jobs are not resumed. Moves already stored are kept.
- [x] 4.2 Video job pipeline persisting moves, events, thumbnails. (R5.9, R7.4)
- [x] 4.3 Routes: health, images, calibration preview, analyses, correction, report. (R1, R3)
- [x] 4.4 Routes: compare, games, move edit/delete, exports. (R4, R6, R7)
- [x] 4.5 Routes: videos, jobs, review. (R5)
- [x] 4.6 API tests: upload validation, analysis (manual), compare outcomes, game revalidation, video job progress/cancel/failure/review with an injected test detector. (R8.4)

## 5. Real inference
- [x] 5.1 `scripts/fetch_reference_weights.py` with explicit AGPL notice and SHA-256 report.
- [~] 5.2 `scripts/verify_real_inference.py` on public validation images; record actual numbers.
  - Numbers are in-distribution (validation frames come from the same videos as training). No real video footage was available, so the real model has not been run through the video workflow.
- [x] 5.3 Real-model pytest marked `real_model`, skipped with reason when weights are absent.

## 6. Frontend
- [x] 6.1 Vite + React + TS scaffold, API client, dev proxy.
- [x] 6.2 Health/settings page and global status bar. (R1)
- [x] 6.3 Shared board component with editing and uncertainty display. (R3.4)
- [x] 6.4 Analyze image page: upload, corner picker, a1 selection, preview, analyze, correct, metadata, FEN. (R3)
- [x] 6.5 Compare page: before source selector, independent calibration, result and candidates. (R4)
- [x] 6.6 Video page: upload, calibration frame, start position, job progress/cancel, review dialog, moves with thumbnails. (R5)
- [x] 6.7 Games page: move list with notation, edit/delete with detached display, exports. (R6, R7)
- [~] 6.8 `npm run build` and `tsc --noEmit` pass.
  - Both pass. The built UI is served by FastAPI (checked with HTTP requests). No automated browser test exists; the UI has not been clicked through in a browser by the agent.

## 7. Delivery
- [x] 7.1 README: setup, commands, dependency versions, limitations, actual test results.
- [x] 7.2 `THIRD_PARTY_NOTICES.md`.
- [x] 7.3 Delete the temporary reference clone; record in README.
