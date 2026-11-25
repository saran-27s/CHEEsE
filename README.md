# chess-vision

A local web app that reads chess positions from photos and moves from recorded video. It runs on your own machine: FastAPI backend, React frontend, SQLite and local files, one background worker. It makes no cloud calls and uses no paid APIs or hosted LLMs. Model weights are downloaded only when you run the fetch script yourself.

What it does:

- **Analyze image.** Upload a photo, click the four board corners, mark a1, preview the rectified board, then run piece recognition with the model, or enter the pieces by hand. Correct any square, set side to move, castling and en passant, and get a FEN that labels every assumed field.
- **Compare.** Detect a single move between a trusted "before" position (standard start, a FEN, or a confirmed before image) and an "after" image. No move is forced: the result is `confirmed`, `no_move`, `ambiguous` or `invalid`, with the ranked candidates.
- **Analyze video.** Upload a recording, calibrate on one frame, choose the starting position, and let the worker extract moves. The board is re-located on every frame, so a nudged board or camera does not break the calibration. A move is recorded only after the board has been unchanged for a set time and legal moves explain the change. When the evidence is unclear, the job pauses and asks you.
- **Results and correction.** Edit or delete any move; later moves are replayed and the ones that no longer follow are detached rather than patched over. Export PGN, FEN with metadata, and JSON reports. Video moves carry the time the board settled and a thumbnail.

Every move is shown as SAN (`Nf3`), UCI (`g1f3`), coordinates (`g1 → f3`) and English ("White moves the knight from g1 to f3."), plus capture, castling, promotion, check and checkmate flags.

When no piece model is installed, the app says so and runs in **manual mode**: calibration, manual entry, compare against your own entries, games and exports all work, and video analysis is disabled. Inference is never faked.

## Quick start (Windows, PowerShell)

From the repository root, the full install and first run:

```powershell
# 1. Backend
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# 2. Optional, needed for video analysis: model runtime and weights (AGPL-3.0, see below)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe scripts\fetch_reference_weights.py --accept-agpl

# 3. Frontend
cd ..\frontend
npm install
npm run build

# 4. Run (single server)
cd ..\backend
.\.venv\Scripts\python.exe -m uvicorn chessvision.main:app --port 8000
```

Open http://127.0.0.1:8000 and check the **Health** tab: it should say "Automated recognition available". Then try the sample video as described in [Sample video](#sample-video).

The sections below explain each step.

## Requirements

Tested on Windows 11 with Python 3.13.15, Node 24.19 and npm 11.17, CPU only (no GPU).

Backend versions installed in the dev venv (`backend/requirements.lock.txt` has the full list):

| Package | Version | | Package | Version |
|---|---|---|---|---|
| fastapi | 0.165.0 | | numpy | 2.5.5 |
| starlette | 1.10.0 | | opencv-python | 5.1.0.95 |
| uvicorn | 0.54.0 | | chess (python-chess) | 1.11.2 |
| pydantic | 2.14.4 | | python-multipart | 0.0.34 |
| pytest | 9.2.0 | | httpx | 0.28.1 |

Optional inference stack: ultralytics 8.4.218, torch 2.15.0+cpu, torchvision 0.29.2.

Frontend: react 19.3.1, react-dom 19.3.1, vite 8.3.8, @vitejs/plugin-react 6.3.0, typescript 7.0.2 (exact tree in `frontend/package-lock.json`).

## Setup

PowerShell, from the repository root.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` is the base runtime and `requirements-dev.txt` adds pytest and httpx. To reproduce the exact tested versions, install from `requirements.lock.txt` instead. That file also contains the optional AGPL inference packages.

### Optional: model runtime and weights (AGPL-3.0)

Automated recognition needs Ultralytics and the reference weights. **Both are AGPL-3.0.** Using them locally is fine. If you redistribute them, or offer the app as a network service that uses them, AGPL obligations apply. Skip this section to stay in manual mode.

```powershell
cd backend
# CPU-only PyTorch first, so pip does not pull the much larger CUDA build
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
# Downloads two .pt files (about 180 MB) into backend/models/ and checks their SHA-256
.\.venv\Scripts\python.exe scripts\fetch_reference_weights.py --accept-agpl
```

The script refuses to run without `--accept-agpl`. Expected hashes are in `backend/models/models.json`:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `yolo11m_pieces.pt` (pieces + `Hand`) | 121,113,130 | `e7467e5e6b7e0ec05e6987c74b8836a7a3c98ff9bdead071a2d0d9867cba4125` |
| `yolo11s_pose_chessboard.pt` (corners) | 58,920,942 | `a847530a214b8ba4e76bd2581d0873297db420fd6f14ec0431bc0cbba3a0e15d` |

Weights are git-ignored and never downloaded automatically. The **Health** tab shows each model's status and the reason when one is unavailable.

### Frontend

```powershell
cd frontend
npm install
npm run build        # runs tsc --noEmit, then vite build into frontend/dist
```

## Running

**Single server (recommended).** Build the frontend once, then:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn chessvision.main:app --port 8000
```

Open http://127.0.0.1:8000. FastAPI serves `frontend/dist` when it exists. The API lives under `/api`. Uvicorn binds to 127.0.0.1 by default. Keep it that way: the app has no authentication.

**Development (two terminals).** Run uvicorn as above, and in `frontend/` run `npm run dev`, then open http://localhost:5173. Vite proxies `/api` to 127.0.0.1:8000.

### Configuration

Environment variables, read at startup:

| Variable | Default | Meaning |
|---|---|---|
| `CHESSVISION_DATA_DIR` | `backend/data` | SQLite database and uploaded or generated files |
| `CHESSVISION_MODELS_MANIFEST` | `backend/models/models.json` | Model manifest |
| `CHESSVISION_WARP_SIZE` | 640 | Rectified board size in pixels |
| `CHESSVISION_MAX_IMAGE_MB` | 25 | Image upload limit |
| `CHESSVISION_MAX_VIDEO_MB` | 2048 | Video upload limit |
| `CHESSVISION_VIDEO_SAMPLE_FPS` | 5.0 | Default frames sampled per second |
| `CHESSVISION_MOTION_FRACTION` | 0.02 | Motion-proxy threshold (only for models without a hand class) |

Resolver, observation and tracker thresholds are code defaults in `backend/chessvision/config.py`. The current values appear on the Health tab. Stability time, minimum frames and sample rate can also be set per video job in the UI.

## How to use it

1. **Calibration** (every image, and once per video). Click the four outer corners of the playing area clockwise as they appear on screen, starting top-left, then choose which clicked corner is a1. Check the preview: a8 must be top-left. Set **Camera position** if the automatic estimate is wrong. Use "Top-down / already rectified" for flat-on or pre-cropped images. The corner model, when installed, can pre-fill corners, but you always confirm them.
2. **Position metadata.** Side to move, castling rights and en passant cannot be read from a photo. Fields you leave blank get defaults that are labelled as assumptions in the FEN export. Castling rights are never inferred from where the kings and rooks stand.
3. **Video review.** When the job pauses, pick a candidate, pick one of the several-move explanations, type one or more UCI moves in order (for example `g1f3 b8c6`), choose "No move happened" (for example a piece was nudged back), or stop. Enter every move visible on the settled frame since the last recorded move. Cancelling or stopping keeps the moves already recorded. If the video ends while a change is still unconfirmed, the job pauses once more so you can enter the last moves.
4. **Exports** (Results and correction tab): PGN download, FEN plus provenance and assumptions as JSON, and a full JSON game report. Each image analysis also has a JSON report. Endpoints: `/api/games/{id}/export/pgn`, `/api/games/{id}/export/fen[?ply=N]`, `/api/games/{id}/export/json`, `/api/analyses/{id}/report`.

Video timestamps mark when the board **settled** (the first frame of the accepted stable window), not when the piece was touched. Moves you enter in a review carry the time of that review.

## Sample video

`samples/sample-game.mp4` is a 42.5 s phone recording (440×640, 30 fps) of a game opening on a real board, seen from above. It is the video used to find and verify the fixes in [How video analysis works](#how-video-analysis-works).

1. Open the **Analyze video** tab and upload `samples/sample-game.mp4`.
2. Calibrate on the frame at 0.00 s. Click the corners clockwise from the top-left of the image, at about (49, 169), (401, 169), (393, 510) and (47, 503). Then pick the **third** clicked corner (bottom-right on screen) as a1. In the preview, a8 must be top-left with the black pieces on the top two rows.
3. Keep the standard starting position and press **Start analysis**. Leave the settings empty for the defaults (5 frames per second), or set 1 frame per second.
4. Answer the review pauses with the moves visible on the settled frame.

The actual moves, checked by hand frame by frame: `1. d4 e5 2. Nf3 e4 3. Nfd2 Nc6 4. c3 d6 5. e3 Nf6 6. Na3`.

Measured on 2026-11-25 through the HTTP API, CPU only, answering reviews with the hand-checked moves:

| Sample rate | Correct moves | Recorded automatically | Entered in reviews | Reviews | Wrong moves | Run time |
|---|---|---|---|---|---|---|
| 5 fps (default) | 11 of 11 | 6 (d4, e5, Nf3, e4, e3, Nf6) | 5 | 2 (30.0 s, 41.2 s) | 0 | about 2 min |
| 1 fps | 11 of 11 | 4 (d4, e5, e3, Nf6) | 7 | 3 (24.0 s, 30.0 s, end of video) | 0 | 45 s |

Reviews happen in this video because Nd2, Nc6, c3 and d6 are played in quick succession with a hand over the board almost the whole time, and the model reads some pieces on this board wrongly (see below). Nothing in that stretch is guessed: the job pauses and asks.

## How video analysis works

The first real-video run of this app recorded 0 moves on the sample video and reported "completed". These were the causes, measured on the sample video, and the fixes:

| Problem | Cause | Fix |
|---|---|---|
| No stable window, so no move was ever validated | The board counted as stable only while all 64 squares read exactly the same. Real detections flicker by a few squares per frame: the longest identical run was 0.33 s, and 1.0 s is needed. | Squares are compared by occupancy and colour (empty / white / black). A window tolerates one flickering square and is matched against its majority view. |
| Starting position rejected | On this board the model swaps king and queen (d1/e1, d8/e8) and misreads bishops and knights with 0.87–0.94 confidence. The exact-label resolver called the standard start `invalid`. | For model input, the resolver scores occupancy and colour. Piece type only breaks ties between moves with the same square changes, such as promotions. Manual entries are still compared by exact piece. |
| Pieces mapped to the wrong squares later in the video | The board slides by up to 27 px (about half a square) during the game, while the calibration stayed fixed. | Each frame is registered to the calibration frame with SIFT features and RANSAC, and the corners move with the board. The job records how many frames were registered and the largest shift. |
| Several moves between two still views | Fast play leaves no stable board between moves, so a single move cannot explain the change. | The tracker searches sequences of up to 3 legal moves. A sequence is recorded only if it is unique, beats the next explanation by the margin, and short views between the moves show which order they were played in. Otherwise the job pauses with the sequences as suggestions. Reviews accept several moves. |
| Persistent misreads blocked every later move | A piece the model always misreads (here a knight on d2 seen as empty) contradicted every candidate. | After a confirmed move or review, squares that still disagree are remembered as known misreads (up to 4) and carry no evidence until the reading changes. Learning is skipped when the disagreement looks like an unentered move. |
| 1 fps found only 2 moves | The stable window needed at least 3 frames as well as 1.0 s. At 1 fps that is 2 s without a hand, which this video almost never has. | When you leave "minimum frames" empty, it becomes the number of frames that span the stability time at the chosen rate (2 at 1 fps, 3 at 5 fps). A value you enter is used as given. |
| Last move lost, job still "completed" | The video ended in `SETTLING`, with no warning. | At the end of the video, an unconfirmed change raises one last review. Every job also gets a summary event with counts and warnings (for example no moves found, or a first frame that disagrees with the starting position). |

Settings for these behaviours are code defaults in `backend/chessvision/config.py`: `tracker.window_tolerance`, `tracker.deviation_squares`, `tracker.follow_board`, `tracker.max_plies`, `tracker.auto_multi_ply` and `resolver.video_match`.

## Tests and verification

### Backend tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -rs -p no:warnings
```

Last run (2026-11-25): **101 passed in 53.8 s**, 0 skipped, 0 failed.

| File | Tests | Covers |
|---|---|---|
| `test_calibration.py` | 20 | all four a1 orientations, every validation rule, camera-side estimate and override |
| `test_notation_metadata.py` | 15 | SAN/UCI/coordinates/English, flags, metadata provenance, castling validation |
| `test_resolver_game.py` | 16 | confirmed / no-move / ambiguous / invalid, promotions, en passant, castling, revalidation and detaching |
| `test_temporal.py` | 18 | state machine, time-based stability, occlusion, dropouts, review handling, move sequences, end-of-video review, learned misreads |
| `test_api.py` | 21 | uploads and validation, manual analysis, compare, games, edits, exports, health |
| `test_video_jobs.py` | 10 | video job progress, cancel, failure, review pause and resume, thumbnails |
| `test_real_model.py` | 1 | the real reference piece model through the API (skipped with a reason if the weights or Ultralytics are missing) |

Except for `test_real_model.py`, all tests use **synthetic fixtures**: observations built directly from known boards, and rendered boards and videos where pieces are coloured discs and a "hand" is a magenta rectangle. An injected `ColorCodeTestDetector` reads those colours back; the real model is not involved. Fixture modules are named `synthetic_board.py` and `fixtures.py` and are labelled synthetic in their docstrings. The only warning is a `StarletteDeprecationWarning` about `httpx` in Starlette's TestClient.

### Frontend checks

```powershell
cd frontend
npm run typecheck    # tsc --noEmit, 14 source files: passes
npm run build        # passes; 262 kB JS (80 kB gzip), 3.3 kB CSS
```

With `uvicorn` running, `/` returned the built `index.html`, the JS bundle was served, and `/api/health` reported automated mode on CPU with the worker alive. **No automated UI tests exist, and the UI has not been clicked through in a browser.**

### Real-model verification

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\verify_real_inference.py --pieces 24 --corners 12
```

The script downloads public validation images (CC BY 4.0) into `backend/verification/` (git-ignored), runs the real models on CPU, and writes `verification/results.json`. Measured on 2026-11-24:

| Model | Images | Result |
|---|---|---|
| Pieces (`yolo11m_pieces.pt`) | 24 | Box precision 1.000 and recall 1.000 at IoU 0.5, same class (616 TP, 0 FP, 0 FN). Model class names match the dataset's `data.yaml`. |
| Pieces, square level | 24 | Agreement between the board built from predictions and the board built from labels: mean 0.999, min 0.984. 22 boards agree on every compared square, 1 board differs on 1 of 64 squares, 1 board had 0 comparable squares (hand boxes cover the compared area). |
| Pieces, latency | 24 | 0.556 s per image mean on CPU; the first call includes a ~4 s warm-up. |
| Corners (`yolo11s_pose_chessboard.pt`) | 12 | Found on 12 of 12. Corner error on 640 px images: mean 4.88 px, median 4.38 px, max 10.74 px. 0.25 s per image. |

**Caveat.** These validation images are random frames from the same videos used for training, so the numbers are in-distribution. They show the integration works and the class mapping is right. They do **not** show accuracy on your board, pieces, lighting or camera.

## Limitations

- **Real video is verified on one recording only.** The sample video above is the only real footage tested. Other boards, pieces, lighting and camera angles are untested.
- **Perspective and piece height.** The homography is exact only for the board plane. Tall pieces lean away from the camera, so pieces are anchored with a heuristic shift toward the camera. At steep angles, neighbouring squares can be confused. Use a high, steep camera, and set Camera position manually if the estimate is wrong.
- **Mirrored images** (selfie cameras) are not supported. The rectified board comes out with its files reversed, and there is no option to un-mirror it.
- **Occlusion.** With the reference model, occlusion comes from its `Hand` class. A hand or arm the model does not box is not treated as occlusion. Models without a hand class fall back to a motion proxy, labelled "degraded", which misses a still hand and reacts to lighting changes.
- **Persistent misdetection** (for example a piece the model always misreads) causes a review pause. After you answer "No move happened", that exact board reading is ignored. A different misreading pauses again.
- **Several moves between stable views** are recorded automatically only when one sequence of up to 3 moves clearly explains the board and its order is visible. Otherwise the job pauses and you enter the moves.
- **Colour-level matching** means a piece-type change with no colour or occupancy change, such as a promotion, is decided by the model's piece labels only.
- **Timestamps** assume constant frame rate (frame index ÷ fps). Variable-frame-rate recordings, such as many phone videos, can drift. The video info shows `timestamp_source`.
- **One worker.** Video jobs run one at a time and others queue. Restarting the backend marks running, queued and paused jobs as failed ("interrupted by backend restart"). Moves already stored are kept, but jobs do not resume.
- **Live webcam** is not implemented. Frame sources sit behind a small interface (`VideoFileFrameSource`) so a camera source can be added later.
- **The model was trained on a narrow set** of boards and cameras (frames from a few videos plus extra images). Accuracy elsewhere is unknown; see the caveat above.
- **No authentication.** Run it on localhost only.

## Project layout

```
backend/
  chessvision/
    core/        calibration, coords, observation, metadata, notation, resolver, temporal tracker, game replay (no I/O)
    inference/   model registry and manifest, optional Ultralytics adapter, motion occlusion proxy
    media/       image validation, video probe and frame source
    jobs/        single background worker and the video job
    storage/     SQLite helpers
    api/         FastAPI routes
  models/        models.json (weights are git-ignored)
  scripts/       fetch_reference_weights.py, verify_real_inference.py
  tests/
frontend/src/    api.ts, components/, pages/, App.tsx
samples/         sample-game.mp4 (real recording used for verification)
docs/reference-audit.md
.kiro/specs/chess-vision/   requirements.md, design.md, tasks.md
THIRD_PARTY_NOTICES.md
```

Stored data lives in `backend/data/`: `chessvision.db` plus `uploads/`, `videos/`, `previews/` and `thumbnails/`. Delete that folder to reset the app.

## Licensing

This project's own code has no license file yet. It imports python-chess, which is GPL-3.0, so if you distribute it, choose a GPL-3.0-compatible license. The optional Ultralytics runtime and the reference weights are AGPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Reference project

The design was informed by an audit of https://github.com/SurawutK/chess-move-tracking-yolo11 at commit `328df61a1bb17bdf816bb2336c22f28d69cd1f64`. The repository has no LICENSE file (the GitHub license API returned 404), only an "MIT" badge in its README. Because that is not a clear grant, **no code was copied**: everything here was written independently. Only the published model weights are used, under their AGPL-3.0 model card, and only through the opt-in fetch script. Findings are in [docs/reference-audit.md](docs/reference-audit.md).

### Reference clone cleanup

The audit used a temporary clone at `C:\Users\HP\AppData\Local\Temp\chessvision-ref-clone` (`%TEMP%\chessvision-ref-clone`). On 2026-11-24 at 17:20 local time, before deletion, `git rev-parse HEAD` returned `328df61a1bb17bdf816bb2336c22f28d69cd1f64` and the origin was the URL above. The clone was then removed with:

```powershell
Remove-Item -LiteralPath (Join-Path $env:TEMP "chessvision-ref-clone") -Recurse -Force
```

`Test-Path` returned `False` afterwards. Nothing else was deleted. No files from the clone were copied into this repository.
