# Reference audit: chess-move-tracking-yolo11

| Item | Value |
|---|---|
| Repository | https://github.com/SurawutK/chess-move-tracking-yolo11 |
| Inspected commit | `328df61a1bb17bdf816bb2336c22f28d69cd1f64` (2025-12-17, "Fix formatting in README for pipeline command") |
| History | 4 commits, first on 2025-12-16 |
| Clone location | `%TEMP%\chessvision-ref-clone` (temporary; deleted after this audit, see the end of this file) |
| Audit date | 2026-11-24 |
| Model repo | https://huggingface.co/surawut/chess-move-tracking-yolo11 (HF API `sha` `c1d115574284b0ca3a1c55dcea51e6baa456ebd0`) |
| Dataset repos | `surawut/chess-pieces-dataset-yolo` (sha `764dfdfdacbce90c4a325b37cc7bfb91dff1e799`), `surawut/chessboard-dataset-yolo` (sha `415e5ab26df0e8c2b088ac62ed7f30ee91fcf481`) |

## 1. What exists

Files at the inspected commit (excluding `.git`):

| Path | Size (bytes) | Content |
|---|---|---|
| `README.md` | 5029 | Project description, claimed metrics, install steps, weight links |
| `requirement.txt` | 113 | Unpinned: chess, huggingface-hub, matplotlib, numpy, opencv-python, pandas, PyYAML, torch, ultralytics, gdown, editdistance |
| `utils/board_localizer.py` | 4978 | YOLO-pose corner detection and perspective warp |
| `utils/chess_state_analyzer.py` | 13567 | Detection-to-square mapping, frame-count stability, square diff |
| `utils/chess_pgn_generator.py` | 8004 | python-chess wrapper: push move, PGN export |
| `utils/fen_utils.py` | 3211 | Board dict to FEN, including castling-rights inference |
| `utils/pipeline.py` | 5201 | CLI: video in, PGN out |
| `chess_move_tracking.ipynb` | 1205961 | Notebook repeating the same classes, plus dataset download, validation, scoring cells (outputs embedded) |
| `output/*.pgn` | 6 files | PGN outputs the author produced for six videos (videos not included) |
| `.gitignore` | 402 | Ignores `*.pt`, `*.mp4`, datasets, and four training/upload scripts that are therefore **not** in the repo |

There is no `LICENSE` file. `GET https://api.github.com/repos/SurawutK/chess-move-tracking-yolo11/license` returned HTTP 404. The README shows an "MIT" badge, but no license text, copyright line, or grant exists in the repository.

## 2. What I actually ran

- `git clone` of the repository and `git log`.
- Read every Python source file and the README in full.
- Extracted and read the notebook's code and markdown cells with a short Python script (outputs skipped).
- Queried the Hugging Face API for the model repo and both dataset repos (file listings, license tags, card text) and fetched `data.yaml` of the pieces dataset.
- Checked `pathlib.Path` behaviour on the URL strings used by `pipeline.py` (see 3.7).

I did **not** run `pipeline.py` or the notebook. The test videos are linked from a Google Drive CSV that the notebook downloads; they are not in the repository, and I did not fetch them. None of the metrics in the README (pose mAP50-95 0.995, detection mAP50 0.978, PGN edit-distance score 0.93) were reproduced. Real-inference checks done later for this project are reported in the project README, not here.

## 3. How the reference works (source-level findings)

### 3.1 Board calibration and perspective
- A YOLO11s-pose model predicts one `chessboard` instance with four keypoints in a fixed semantic order: `a1, h1, a8, h8`. Keypoints are labelled by chess meaning, not by image position, so the model is meant to handle rotated boards.
- Calibration runs on frames until it succeeds, then **locks** the 3x3 matrix for the rest of the video (`cv2.getPerspectiveTransform`). It rejects a detection if any keypoint coordinate is `<= 0` or if the mean keypoint confidence is below 0.5.
- No geometric validation (convexity, area, self-intersection) is performed.
- Destination layout in the 640x640 warp: `a8` top-left, `h8` top-right, `a1` bottom-left, `h1` bottom-right (White at the bottom).

### 3.2 Piece and hand detection
- YOLO11m detector, 13 classes, run on the warped 640x640 image with `conf=0.25`.
- Class order from `data.yaml`: `['Hand', 'bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']`. Code reads names from the model, not from this list.
- Preprocessing documented in the notebook/dataset: EXIF auto-orient, stretch-resize to 640x640. The training images are already-warped boards, so the detector expects warped input.
- Any `Hand` detection anywhere marks the whole frame unusable.

### 3.3 Piece-to-square mapping and anchors
- Each box maps to a square through a single point: horizontal centre, and `y1 + 0.25 * height` (a point near the top of the box). The code comment calls this "more robust to perspective distortion".
- Coordinates are clamped to 0..7, so detections outside the board are silently assigned to edge squares.
- If two boxes land on one square, the higher-confidence box wins; the other is dropped without trace.
- Observation: whether the top or bottom of a box is the right anchor depends on which side of the board the camera was on (tall pieces lean away from the camera after warping). The fixed 0.25-from-top rule matches the author's camera set-up; it is not a general rule.

### 3.4 Temporal stability and state transitions
- Each frame's board dict is serialised to a string. A state is accepted after it stays identical for `stability_thresh` consecutive frames. `pipeline.py` passes `1.5 * 30 = 45`, i.e. it assumes 30 fps; the threshold is a frame count, not a time.
- Frames with a hand or fewer than 2 pieces reset the counter.
- Initialisation: the first stable state becomes the initial position; the side to move is taken from the colour of the first piece that moves.

### 3.5 Move inference
- `_detect_move_diff` scans changed squares: a square that lost its piece becomes `from`, any square that has a piece in the new state becomes `to`. With more than two changed squares (castling: 4, en passant: 3), the result depends on Python set iteration order.
- The analyzer updates its "previous stable state" to the new raw state **before** the move is validated. If python-chess then rejects the move, the analyzer and the game record disagree from then on.
- The first move is pushed with `ignore_rules=True`, so an illegal first move can enter the record.

### 3.6 Legal-move validation, PGN and special moves
- python-chess is the referee: `Move.from_uci(from+to)`, checked against `board.legal_moves`, exported with `chess.pgn`.
- Promotion is always to a queen.
- Castling works only if the two-square diff happens to pick the king's squares.
- `fen_utils.generate_fen_from_dict` returns full castling rights `KQkq` for the exact start placement and otherwise **infers castling rights from king/rook placement**, with en passant `-` and counters `0 1`.
- PGN "clean format" strips the result token and replaces `...` with `.`, which is not valid PGN movetext for Black-first games.

### 3.7 Other defects found by reading
- `pipeline.py` builds `Path("https://huggingface.co/...")`. Verified locally: on Windows this becomes `https:\huggingface.co\...` and on POSIX `https:/huggingface.co/...`, so the auto-download claimed by the README is unlikely to work from the CLI (the notebook passes plain strings and is not affected).
- `requirement.txt` is unpinned; the README says `requirements.txt`.

## 4. Useful architectural ideas (high level only)

1. Rectify the board first, then detect pieces in the canonical top-down view.
2. Treat a hand class as an occlusion signal that suspends state changes.
3. Require a period of stability before accepting a new board state.
4. Use a chess rules engine to reject impossible transitions.
5. Allow a non-standard starting position.

Weaknesses this project will not repeat: frame-count stability, raw detections overwriting confirmed state, set-order-dependent diffing, castling-rights inference from placement, auto-queen promotion, unchecked first move, silent clamping of out-of-board boxes.

## 5. Models and assets

| Asset | Where | Status |
|---|---|---|
| `models/yolo11m_pieces.pt` | HF `surawut/chess-move-tracking-yolo11` | Listed by the HF API; not in the GitHub repo |
| `models/yolo11s_pose_chessboard.pt` | same | Listed by the HF API; not in the GitHub repo |
| Total model repo storage | HF API `usedStorage` | 180,034,072 bytes |
| Pieces dataset (warped 640x640, YOLO boxes) | HF `surawut/chess-pieces-dataset-yolo` | Public, includes `valid/` split |
| Corner dataset (YOLO pose) | HF `surawut/chessboard-dataset-yolo` | Public |
| Test videos and solution CSV | Google Drive links inside the notebook | Not inspected, not downloaded |
| Training scripts | Listed in `.gitignore` | Missing from the repo |

Hand detection exists only as the `Hand` class of the pieces model. There is no separate hand model.

## 6. Licensing and reuse restrictions

| Component | License evidence | Consequence |
|---|---|---|
| GitHub source code | No LICENSE file, GitHub license API 404; README badge says MIT without text | **Unclear.** Treated as all rights reserved. No code is copied, translated, or adapted. |
| Model weights (HF) | Model card `license: agpl-3.0`; fine-tuned from Ultralytics YOLO11 | AGPL-3.0. Fine for local use. Redistributing the weights, or offering the app as a network service that uses them, triggers AGPL obligations. Weights are not committed to this project. |
| Ultralytics library | AGPL-3.0 (Ultralytics' published license; enterprise license sold separately) | Kept as an **optional** dependency behind an adapter. Not installed by the base requirements. |
| Pieces and corner datasets | HF card `license: cc-by-4.0`; `data.yaml` names Roboflow workspace `chess-detection-8s15a`, project `chess-pieces-detection-1yb2h`, version 4, CC BY 4.0 | Attribution required if images are redistributed. Validation images are used only as local, uncommitted verification inputs. |
| python-chess | GPL-3.0 | Used as a library by this project's backend; see THIRD_PARTY_NOTICES.md. |

## 7. Reuse vs independent implementation

- **Reused:** nothing from the GitHub source. The published model weights may be used through an optional adapter, downloaded by the user under AGPL-3.0, with the class names and the 640x640 warped-input convention taken from the published model card and dataset `data.yaml` (facts about the asset, not code).
- **Independently implemented:** calibration with geometric validation and explicit a1 orientation; camera-aware, model-configurable anchoring; per-square probabilistic observations with uncertainty and occlusion; time-based (not frame-count) stability; a legal-successor resolver that scores no-move and every legal move including all promotion choices, with margin and coverage checks; a confirmed-game model separate from observations, with downstream revalidation; deterministic SAN/UCI/English generation; explicit FEN metadata without placement-based castling inference.

## 8. Actual blockers

1. Source license is unclear, so no source reuse (resolved by independent implementation).
2. Weights are AGPL-3.0. Real automated recognition requires the user to install the optional `ultralytics` dependency and download the weights. Without them the app runs in manual mode only.
3. No reference test videos were available locally, so the video workflow cannot be validated end to end on real footage from the reference project.
4. The reference detector was trained on a narrow set of boards and camera set-ups (frames from five videos plus extra images). Accuracy on other boards is unknown.

## 9. Clone removal

Recorded in section "Reference clone cleanup" of the project README once the temporary clone is deleted.
