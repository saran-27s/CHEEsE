"""Real-model end-to-end tests. Skipped (with the reason) unless the AGPL-3.0 weights, the optional
ultralytics runtime, and the validation images fetched by scripts/verify_real_inference.py exist."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chessvision.api.app import create_app
from chessvision.config import BACKEND_DIR, load_settings

MODELS = BACKEND_DIR / "models"
VERIFY = BACKEND_DIR / "verification" / "surawut__chess-pieces-dataset-yolo" / "valid"
PIECE_NAMES = ['Hand', 'bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']
TO_FEN = {"wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
          "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k"}


def _skip_reason() -> str | None:
    if importlib.util.find_spec("ultralytics") is None:
        return "optional runtime 'ultralytics' not installed"
    if not (MODELS / "yolo11m_pieces.pt").exists():
        return "piece weights not downloaded (scripts/fetch_reference_weights.py --accept-agpl)"
    if not VERIFY.exists() or not any((VERIFY / "images").glob("*.jpg")):
        return "validation images not fetched (scripts/verify_real_inference.py)"
    return None


pytestmark = [pytest.mark.real_model, pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")]


def _label_placement(label: Path) -> tuple[dict[str, str], bool]:
    """Square placement from YOLO labels using the box-centre rule; second value = hand present."""
    placement, hand = {}, False
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        name = PIECE_NAMES[int(parts[0])]
        if name == "Hand":
            hand = True
            continue
        cx, cy = float(parts[1]), float(parts[2])
        f, row = int(cx * 8), int(cy * 8)
        placement["abcdefgh"[f] + str(8 - row)] = TO_FEN[name]
    return placement, hand


def test_real_piece_model_through_api(tmp_path):
    settings = load_settings(data_dir=tmp_path / "data")  # real manifest and real weights
    with TestClient(create_app(settings, serve_frontend=False)) as client:
        health = client.get("/api/health").json()
        assert health["mode"] == "automated", health["mode_label"]
        assert health["piece_detector"]["id"] == "surawut-yolo11m-pieces"
        assert health["piece_detector"]["injected"] is False
        checked = 0
        for img in sorted((VERIFY / "images").glob("*.jpg")):
            placement, hand = _label_placement(VERIFY / "labels" / (img.stem + ".txt"))
            if hand:
                continue  # compare only unoccluded boards here
            r = client.post("/api/images", files={"file": (img.name, img.read_bytes(), "image/jpeg")})
            assert r.status_code == 200, r.text
            image_id = r.json()["id"]
            # The dataset images are already warped: corners = image corners, a1 bottom-left.
            a = client.post("/api/analyses", json={"image_id": image_id, "a1_index": 3, "mode": "auto",
                                                   "corners": [[0, 0], [640, 0], [640, 640], [0, 640]]}).json()
            assert a["mode"] == "model" and a["model"]["id"] == "surawut-yolo11m-pieces"
            seen = a["position"]["placement"]
            squares = set(seen) | set(placement)
            agree = sum(1 for sq in squares if seen.get(sq) == placement.get(sq))
            assert agree >= len(squares) - 1, (img.name, seen, placement)
            checked += 1
            if checked >= 3:
                break
        assert checked > 0, "no hand-free validation images available"
