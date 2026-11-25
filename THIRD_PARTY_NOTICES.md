# Third-party notices

This file lists the third-party code, model weights and datasets that chess-vision uses, grouped by type. They are licensed separately. Licenses for Python packages were read from the installed package metadata (`importlib.metadata`), and licenses for npm packages from each package's `package.json`, on 2026-11-24. Check the upstream project before relying on this list for a distribution.

Nothing from these projects is vendored into this repository. Dependencies are installed by pip or npm. Weights and dataset images are downloaded only by the scripts described below and are git-ignored.

## 1. Code: backend runtime (always installed)

| Package | Version | License | Notes |
|---|---|---|---|
| python-chess (`chess`) | 1.11.2 | **GPL-3.0-or-later** | Move generation, legality, SAN, FEN, PGN. https://github.com/niklasf/python-chess |
| FastAPI | 0.165.0 | MIT | https://github.com/fastapi/fastapi |
| Starlette | 1.10.0 | BSD-3-Clause | |
| Uvicorn | 0.54.0 | BSD-3-Clause | |
| Pydantic / pydantic-core | 2.14.4 / 2.50.5 | MIT | |
| NumPy | 2.5.5 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | |
| opencv-python | 5.1.0.95 | Apache-2.0 | OpenCV is Apache-2.0 from version 4.5.0. The wheel bundles further third-party libraries; see its own notices. |
| python-multipart | 0.0.34 | Apache-2.0 | |
| AnyIO, h11, annotated-types | 4.16.0, 0.16.0, 0.8.0 | MIT | Transitive |
| Click | 8.5.1 | BSD-3-Clause | Transitive (uvicorn) |
| typing-extensions | 4.16.0 | PSF-2.0 | Transitive |

**GPL-3.0 note.** The backend imports python-chess. Anyone who distributes this backend, with or without changes, must do so under terms compatible with GPL-3.0. Running it locally carries no extra obligation.

## 2. Code: development and test only

| Package | Version | License |
|---|---|---|
| pytest | 9.2.0 | MIT |
| HTTPX | 0.28.1 | BSD-3-Clause |

## 3. Code: optional inference runtime (`requirements-inference.txt`)

Installed only when the user opts in to automated recognition.

| Package | Version | License | Notes |
|---|---|---|---|
| Ultralytics | 8.4.218 | **AGPL-3.0** | YOLO runtime. https://github.com/ultralytics/ultralytics. Ultralytics also sells an enterprise license. |
| ultralytics-thop | 2.2.5 | AGPL-3.0 | Transitive |
| ultralytics-platform | 0.1.121 | AGPL-3.0-only | Transitive |
| PyTorch (`torch`) | 2.15.0+cpu | BSD-3-Clause, with bundled components under Apache-2.0 (incl. LLVM exception), BSD-2-Clause, BSL-1.0, ISC, MIT, Zlib | https://github.com/pytorch/pytorch |
| torchvision | 0.29.2 | BSD-3-Clause | |
| Pillow | 13.0.0 | MIT-CMU | Transitive |
| Matplotlib | 3.11.2 | Matplotlib license (PSF-based) | Transitive |
| Polars | 2.1.0 | MIT | Transitive |
| Requests | 2.34.2 | Apache-2.0 | Transitive |
| psutil | 7.2.2 | BSD-3-Clause | Transitive |

`backend/requirements.lock.txt` lists every package and version in the tested environment.

**AGPL-3.0 note.** chess-vision talks to Ultralytics only through an adapter (`backend/chessvision/inference/yolo.py`) and works without it in manual mode. If you distribute the app together with Ultralytics, or let other people use it over a network while Ultralytics is installed, AGPL-3.0 obligations apply, including offering the corresponding source.

## 4. Code: frontend

| Package | Version | License |
|---|---|---|
| React, react-dom | 19.3.1 | MIT |
| scheduler | 0.28.0 | MIT (transitive, bundled into the build) |
| Vite | 8.3.8 | MIT (build tool) |
| @vitejs/plugin-react | 6.3.0 | MIT (build tool) |
| TypeScript | 7.0.2 | Apache-2.0 (build tool) |
| @types/react, @types/react-dom | 19.3.0 | MIT (type definitions) |

Only React, react-dom and scheduler end up in `frontend/dist`. Run `npm ls --all` in `frontend/` for the full development tree.

## 5. Model weights (optional, downloaded on request)

| File | Source | License |
|---|---|---|
| `yolo11m_pieces.pt` (SHA-256 `e7467e5e…cba4125`) | https://huggingface.co/surawut/chess-move-tracking-yolo11 (repo revision `c1d115574284b0ca3a1c55dcea51e6baa456ebd0`) | **AGPL-3.0** (model card). Fine-tuned from Ultralytics YOLO11. |
| `yolo11s_pose_chessboard.pt` (SHA-256 `a847530a…3a0e15d`) | same repository | **AGPL-3.0** (model card). Fine-tuned from Ultralytics YOLO11. |

Author: Surawut (Hugging Face user `surawut`). Full hashes are in `backend/models/models.json`. The weights are fetched only by `backend/scripts/fetch_reference_weights.py --accept-agpl`, are git-ignored, and must not be redistributed except under AGPL-3.0.

## 6. Datasets (used only for local verification, not redistributed)

`backend/scripts/verify_real_inference.py` downloads validation images and labels from these datasets into `backend/verification/`, which is git-ignored:

| Dataset | Origin | License |
|---|---|---|
| `surawut/chess-pieces-dataset-yolo` (HF revision `764dfdfdacbce90c4a325b37cc7bfb91dff1e799`) | Roboflow workspace `chess-detection-8s15a`, project `chess-pieces-detection-1yb2h`, version 4 | CC BY 4.0 |
| `surawut/chessboard-dataset-yolo` (HF revision `415e5ab26df0e8c2b088ac62ed7f30ee91fcf481`) | Roboflow workspace `chess-detection-8s15a`, project `chessboard-localization-g58s1`, version 2 | CC BY 4.0 |

Attribution: datasets by the Roboflow workspace `chess-detection-8s15a`, published on Hugging Face by `surawut`, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). If you redistribute any of these images, keep this attribution and indicate any changes.

## 7. Reference project (audited, no code used)

https://github.com/SurawutK/chess-move-tracking-yolo11 at commit `328df61a1bb17bdf816bb2336c22f28d69cd1f64` was read for the audit in `docs/reference-audit.md`. It has no LICENSE file; its README shows an "MIT" badge. Because that is not a clear license grant, **no code, text or notebook content from it was copied** into this project. The temporary clone was deleted after the audit (see README, "Reference clone cleanup").
