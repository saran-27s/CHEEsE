"""API tests. Images are synthetic renders (test fixtures); 'auto' analyses use an explicitly
injected ColorCodeTestDetector, never a silent fallback."""

import chess
import pytest

from api_helpers import make_client
from synthetic_board import A1_INDEX, QUAD, ColorCodeTestDetector, boards, png_bytes, render_photo


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def upload(client, board=None, hand=None, name="board.png"):
    data = png_bytes(render_photo(board or chess.Board(), hand))
    r = client.post("/api/images", files={"file": (name, data, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def analyse(client, image_id, mode="manual", **extra):
    return client.post("/api/analyses", json={"image_id": image_id, "corners": QUAD, "a1_index": A1_INDEX,
                                              "mode": mode, **extra})


def placement_of(board):
    return {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()}


# ---- health -------------------------------------------------------------------
def test_health_reports_manual_mode_and_reasons(client):
    h = client.get("/api/health").json()
    assert h["mode"] == "manual"
    assert h["mode_label"].startswith("Manual mode (no automated recognition)")
    assert h["piece_detector"]["status"] == "missing_file"
    assert h["worker"]["alive"] is True
    assert h["runtime_installed"]["ultralytics"] in (True, False)


def test_health_reports_injected_detector(client):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    h = client.get("/api/health").json()
    assert h["mode"] == "automated" and h["piece_detector"]["injected"] is True
    assert h["occlusion"]["hand_detection"] is True


# ---- upload validation -----------------------------------------------------------
def test_upload_rejects_bad_extension(client):
    r = client.post("/api/images", files={"file": ("x.gif", b"GIF89a....", "image/gif")})
    assert r.status_code == 415


def test_upload_rejects_non_image_content(client):
    r = client.post("/api/images", files={"file": ("x.png", b"not an image at all", "image/png")})
    assert r.status_code == 415 and "not a JPEG, PNG or WebP" in r.json()["detail"]


def test_upload_rejects_extension_mismatch(client):
    data = png_bytes(render_photo(chess.Board()))
    r = client.post("/api/images", files={"file": ("x.jpg", data, "image/jpeg")})
    assert r.status_code == 415 and "does not match" in r.json()["detail"]


def test_upload_rejects_empty_and_oversize(tmp_path):
    with make_client(tmp_path, max_image_mb=1) as c:
        assert c.post("/api/images", files={"file": ("x.png", b"", "image/png")}).status_code == 400
        big = b"\x89PNG\r\n\x1a\n" + b"0" * 1_100_000
        assert c.post("/api/images", files={"file": ("x.png", big, "image/png")}).status_code == 413


def test_upload_rejects_truncated_image(client):
    data = png_bytes(render_photo(chess.Board()))[:200]
    r = client.post("/api/images", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 422


# ---- calibration -----------------------------------------------------------------
def test_calibration_preview_valid_and_invalid(client):
    image_id = upload(client)
    bad = client.post("/api/calibrations/preview",
                      json={"image_id": image_id, "corners": list(reversed(QUAD)), "a1_index": 0}).json()
    assert bad["valid"] is False and any("counter-clockwise" in e for e in bad["errors"])
    good = client.post("/api/calibrations/preview",
                       json={"image_id": image_id, "corners": QUAD, "a1_index": A1_INDEX}).json()
    assert good["valid"] is True
    assert client.get(good["preview_url"]).status_code == 200
    assert good["calibration"]["chess_corners"]["a1"] == list(QUAD[A1_INDEX])


def test_analysis_rejects_invalid_calibration(client):
    image_id = upload(client)
    r = client.post("/api/analyses", json={"image_id": image_id, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]],
                                           "a1_index": 0, "mode": "manual"})
    assert r.status_code == 422


# ---- analyze image -----------------------------------------------------------------
def test_auto_analysis_without_model_is_409_not_faked(client):
    image_id = upload(client)
    r = analyse(client, image_id, mode="auto")
    assert r.status_code == 409 and "Use manual mode" in r.json()["detail"]


def test_manual_analysis_correction_and_fen_assumptions(client):
    image_id = upload(client)
    a = analyse(client, image_id, manual_start="standard").json()
    assert a["mode"] == "manual" and a["mode_label"].startswith("Manual entry")
    assert a["position"]["fen"].endswith(" w - - 0 1")
    assert len(a["position"]["assumptions"]) == 5
    assert "piece-placement evidence only" in a["note"]
    placement = placement_of(chess.Board())
    r = client.put(f"/api/analyses/{a['id']}/correction",
                   json={"placement": placement, "metadata": {"side_to_move": "w", "castling": "KQkq"}}).json()
    assert r["position"]["fen"] == chess.STARTING_FEN
    assert r["position"]["metadata"]["provenance"]["castling"] == "user"
    assert r["position"]["metadata"]["provenance"]["halfmove"] == "assumed_default"
    assert r["observation"]["source"] == "manual"  # the observation is kept unchanged
    # Impossible castling claim is reported, not silently dropped.
    del placement["h1"]
    r2 = client.put(f"/api/analyses/{a['id']}/correction",
                    json={"placement": placement, "metadata": {"side_to_move": "w", "castling": "KQkq"}}).json()
    assert r2["position"]["valid"] is False
    assert r2["correction"]["changed_from_observation"] == ["h1"]


def test_auto_analysis_with_injected_detector(client):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    board = boards("e2e4")[1]
    a = analyse(client, upload(client, board), mode="auto").json()
    assert a["mode"] == "model" and a["model"]["id"] == "test-colorcode-detector"
    assert a["position"]["placement"] == placement_of(board)
    report = client.get(f"/api/analyses/{a['id']}/report").json()
    assert report["observation"]["source"] == "model" and report["input"]["sha256"]


def test_hand_occludes_squares_in_single_image(client):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    a = analyse(client, upload(client, chess.Board(), hand=["e2"]), mode="auto").json()
    obs = {s["square"]: s for s in a["observation"]["squares"]}
    assert obs["e2"]["occluded"] is True and a["observation"]["hand_detected"] is True
    assert "e2" in a["uncertain_squares"]


# ---- compare ---------------------------------------------------------------------
def manual_after(client, board):
    a = analyse(client, upload(client)).json()
    client.put(f"/api/analyses/{a['id']}/correction", json={"placement": placement_of(board)})
    return a["id"]


def test_compare_start_vs_after_confirmed(client):
    after = manual_after(client, boards("g1f3")[1])
    r = client.post("/api/compare", json={"before": {"kind": "start"}, "after_analysis_id": after}).json()
    assert r["status"] == "confirmed"
    d = r["best"]["description"]
    assert (d["san"], d["uci"], d["coordinates"]) == ("Nf3", "g1f3", "g1 → f3")
    assert d["english"] == "White moves the knight from g1 to f3."
    assert r["after"]["evidence_source"] == "user-corrected placement of the after image"


def test_compare_no_move_and_invalid(client):
    same = manual_after(client, chess.Board())
    assert client.post("/api/compare", json={"before": {"kind": "start"}, "after_analysis_id": same}).json()["status"] == "no_move"
    two = manual_after(client, boards("e2e4", "e7e5")[2])
    r = client.post("/api/compare", json={"before": {"kind": "start"}, "after_analysis_id": two}).json()
    assert r["status"] == "invalid" and r["after_fen"] is None


def test_compare_from_fen_and_from_before_analysis(client):
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    b = chess.Board(fen)
    b.push_uci("e1g1")
    after = manual_after(client, b)
    r = client.post("/api/compare", json={"before": {"kind": "fen", "fen": fen}, "after_analysis_id": after}).json()
    assert r["status"] == "confirmed" and r["best"]["description"]["san"] == "O-O"
    # Before = analysis without side to move -> rejected with an explanation.
    before = analyse(client, upload(client)).json()["id"]
    client.put(f"/api/analyses/{before}/correction", json={"placement": placement_of(chess.Board(fen))})
    r2 = client.post("/api/compare", json={"before": {"kind": "analysis", "analysis_id": before},
                                           "after_analysis_id": after})
    assert r2.status_code == 422 and "side to move" in r2.json()["detail"]
    client.put(f"/api/analyses/{before}/correction",
               json={"placement": placement_of(chess.Board(fen)), "metadata": {"side_to_move": "w", "castling": "KQkq"}})
    r3 = client.post("/api/compare", json={"before": {"kind": "analysis", "analysis_id": before},
                                           "after_analysis_id": after}).json()
    assert r3["status"] == "confirmed" and r3["best"]["uci"] == "e1g1"


def test_compare_castling_not_assumed(client):
    # Without user-supplied castling rights the default is '-', so castling cannot be confirmed.
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    before = analyse(client, upload(client)).json()["id"]
    client.put(f"/api/analyses/{before}/correction",
               json={"placement": placement_of(chess.Board(fen)), "metadata": {"side_to_move": "w"}})
    b = chess.Board(fen)
    b.push_uci("e1g1")
    after = manual_after(client, b)
    r = client.post("/api/compare", json={"before": {"kind": "analysis", "analysis_id": before},
                                          "after_analysis_id": after}).json()
    assert r["status"] == "invalid"
    assert any("Castling rights assumed" in a for a in r["before"]["assumptions"])


# ---- games, correction, exports ------------------------------------------------------
def test_game_edit_revalidates_downstream_and_exports(client):
    g = client.post("/api/games", json={"start": {"kind": "start"},
                                        "moves": [{"uci": u} for u in ("e2e4", "e7e5", "g1f3", "b8c6")]}).json()
    assert [m["san"] for m in g["moves"]] == ["e4", "e5", "Nf3", "Nc6"]
    r = client.put(f"/api/games/{g['id']}/moves/1", json={"uci": "g1h3"}).json()
    assert [m["uci"] for m in r["moves"]] == ["g1h3", "e7e5"]
    assert [d["uci"] for d in r["detached"]] == ["g1f3", "b8c6"]
    assert r["revalidation"]["detached_now"][0]["uci"] == "g1f3"
    pgn = client.get(f"/api/games/{g['id']}/export/pgn").text
    assert "1. Nh3 e5" in pgn and "Nf3" not in pgn
    fen = client.get(f"/api/games/{g['id']}/export/fen").json()
    # python-chess only writes an en passant square when a capture is actually possible.
    assert fen["fen"] == "rnbqkbnr/pppp1ppp/8/4p3/8/7N/PPPPPPPP/RNBQKB1R w KQkq - 0 2"
    assert fen["provenance"]["castling"] == "derived_from_game"
    js = client.get(f"/api/games/{g['id']}/export/json").json()
    assert js["moves"][0]["english"] == "White moves the knight from g1 to h3."
    assert client.put(f"/api/games/{g['id']}/moves/2", json={"uci": "e2e4"}).status_code == 422
    d = client.delete(f"/api/games/{g['id']}/moves/1").json()
    assert d["moves"] == [] and d["revalidation"]["detached_now"][0]["uci"] == "e7e5"
    assert client.delete(f"/api/games/{g['id']}/detached").json()["detached"] == []


def test_game_rejects_illegal_initial_moves_and_append(client):
    r = client.post("/api/games", json={"start": {"kind": "start"}, "moves": [{"uci": "e2e5"}]})
    assert r.status_code == 422
    g = client.post("/api/games", json={"start": {"kind": "fen", "fen": "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"}}).json()
    assert '[SetUp "1"]' in g["pgn"]
    assert client.post(f"/api/games/{g['id']}/moves", json={"uci": "a1a8"}).json()["moves"][0]["is_check"] is True
    assert client.post(f"/api/games/{g['id']}/moves", json={"uci": "a1a2"}).status_code == 422


def test_invalid_fen_start_rejected(client):
    r = client.post("/api/games", json={"start": {"kind": "fen", "fen": "8/8/8/8/8/8/8/8 w - - 0 1"}})
    assert r.status_code == 422


def test_file_route_blocks_traversal(client):
    assert client.get("/api/files/previews/..%2Fchessvision.db").status_code == 404
    assert client.get("/api/files/uploads/x.png").status_code == 404
