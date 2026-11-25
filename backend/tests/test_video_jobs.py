"""Video upload and job tests on synthetic videos rendered by tests/synthetic_board.py.

The detector is the injected ColorCodeTestDetector test fixture, not a real model.
"""

import chess
import pytest

from api_helpers import make_client, wait_job
from synthetic_board import A1_INDEX, QUAD, ColorCodeTestDetector, boards, write_video


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def upload_video(client, path):
    with open(path, "rb") as f:
        r = client.post("/api/videos", files={"file": (path.name, f, "video/x-msvideo")})
    assert r.status_code == 200, r.text
    return r.json()


def start(client, video_id, **extra):
    body = {"corners": QUAD, "a1_index": A1_INDEX, "sample_fps": 10, **extra}
    return client.post(f"/api/videos/{video_id}/jobs", json=body)


def two_move_video(tmp_path):
    b0, b1, b2 = boards("e2e4", "e7e5")
    return write_video(tmp_path / "game.avi", [
        (1.5, b0, None), (1.0, b0, ["e2", "e4"]), (2.0, b1, None),
        (1.0, b1, ["e7", "e5"]), (2.0, b2, None)])


def test_video_upload_validation(client, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 2048)
    with open(bad, "rb") as f:
        r = client.post("/api/videos", files={"file": ("bad.mp4", f, "video/mp4")})
    assert r.status_code == 422
    r2 = client.post("/api/videos", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r2.status_code == 415


def test_video_meta_and_frame(client, tmp_path):
    v = upload_video(client, two_move_video(tmp_path))
    assert v["meta"]["fps"] == pytest.approx(10.0)
    assert v["meta"]["duration"] == pytest.approx(7.5, abs=0.15)
    r = client.get(f"/api/videos/{v['id']}/frame", params={"t": 2.0})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    p = client.post(f"/api/videos/{v['id']}/calibration-preview",
                    json={"t": 0.5, "corners": QUAD, "a1_index": A1_INDEX}).json()
    assert p["valid"] and client.get(p["preview_url"]).status_code == 200


def test_job_requires_model(client, tmp_path):
    v = upload_video(client, two_move_video(tmp_path))
    r = start(client, v["id"])
    assert r.status_code == 409 and "piece detector" in r.json()["detail"]


def test_job_rejects_bad_calibration_and_start(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    v = upload_video(client, two_move_video(tmp_path))
    assert client.post(f"/api/videos/{v['id']}/jobs",
                       json={"corners": list(reversed(QUAD)), "a1_index": 0}).status_code == 422
    assert start(client, v["id"], start={"kind": "fen", "fen": "bad fen"}).status_code == 422


def test_job_detects_moves_with_timestamps_and_thumbnails(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    v = upload_video(client, two_move_video(tmp_path))
    job = start(client, v["id"]).json()
    assert job["status"] in ("queued", "running")
    done = wait_job(client, job["id"], {"completed", "failed"})
    assert done["status"] == "completed", done["error"]
    assert done["progress"] == 1.0
    assert done["capabilities"]["hand_detection"] is True
    moves = [e for e in done["events"] if e["kind"] == "move"]
    assert [m["payload"]["uci"] for m in moves] == ["e2e4", "e7e5"]
    # Board settled right after each hand left the board (t = 2.5 s and 5.5 s).
    assert moves[0]["t_settled"] == pytest.approx(2.5, abs=0.11)
    assert moves[1]["t_settled"] == pytest.approx(5.5, abs=0.11)
    assert client.get(moves[0]["thumbnail_url"]).status_code == 200
    game = client.get(f"/api/games/{done['game_id']}").json()
    assert [m["san"] for m in game["moves"]] == ["e4", "e5"]
    assert all(m["provenance"] == "resolver_video" and m["has_evidence"] for m in game["moves"])
    pgn = client.get(f"/api/games/{done['game_id']}/export/pgn").text
    assert "1. e4 { board settled at 2.50 s } 1... e5 { board settled at 5.50 s }" in pgn


def test_degraded_motion_mode_without_hand_class(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector(with_hand=False))
    v = upload_video(client, two_move_video(tmp_path))
    done = wait_job(client, start(client, v["id"]).json()["id"], {"completed", "failed"})
    assert done["status"] == "completed", done["error"]
    assert done["capabilities"]["degraded"] is True
    assert "not hand detection" in done["capabilities"]["note"]
    # The magenta "hand" now looks like missing pieces / motion; it must not create captures.
    game = client.get(f"/api/games/{done['game_id']}").json()
    assert all(not m["is_capture"] for m in game["moves"])


def test_job_failure_is_reported(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector(fail=True))
    v = upload_video(client, two_move_video(tmp_path))
    done = wait_job(client, start(client, v["id"]).json()["id"], {"completed", "failed"})
    assert done["status"] == "failed" and "simulated detector failure" in done["error"]


def test_job_cancellation(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    b0 = chess.Board()
    long_video = write_video(tmp_path / "long.avi", [(60.0, b0, None)], fps=10)
    v = upload_video(client, long_video)
    job = start(client, v["id"]).json()
    wait_job(client, job["id"], {"running"})
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    done = wait_job(client, job["id"], {"cancelled", "completed", "failed"})
    assert done["status"] == "cancelled" and done["progress"] < 1.0
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_review_pause_and_user_resolution(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    client.app.state.cv.settings.tracker.auto_multi_ply = False  # exercise the review path
    b0, b1, b2 = boards("e2e4", "e7e5")
    # Two moves happen during one hand occlusion: not resolved automatically when auto_multi_ply is off.
    path = write_video(tmp_path / "two.avi", [(1.5, b0, None), (1.0, b0, ["e4", "e5"]), (3.0, b2, None)])
    v = upload_video(client, path)
    job = start(client, v["id"]).json()
    paused = wait_job(client, job["id"], {"paused_for_review", "completed", "failed"})
    assert paused["status"] == "paused_for_review"
    review = [e for e in paused["events"] if e["kind"] == "review"][0]
    assert review["payload"]["result"]["status"] == "invalid"
    assert paused["pending_review_event_id"] == review["id"]
    # Illegal choice rejected; wrong event rejected.
    assert client.post(f"/api/jobs/{job['id']}/review",
                       json={"event_id": review["id"], "action": "move", "uci": "e7e5"}).status_code == 422
    assert client.post(f"/api/jobs/{job['id']}/review",
                       json={"event_id": "nope", "action": "no_move"}).status_code == 409
    # The user supplies the first move; the tracker then confirms the second from the evidence.
    ok = client.post(f"/api/jobs/{job['id']}/review", json={"event_id": review["id"], "action": "move", "uci": "e2e4"})
    assert ok.status_code == 200
    done = wait_job(client, job["id"], {"completed", "failed"})
    assert done["status"] == "completed", done["error"]
    game = client.get(f"/api/games/{done['game_id']}").json()
    assert [(m["uci"], m["provenance"]) for m in game["moves"]] == [("e2e4", "user"), ("e7e5", "resolver_video")]


def test_review_stop_cancels_and_keeps_moves(client, tmp_path):
    client.app.state.cv.registry.inject("piece_detector", ColorCodeTestDetector())
    client.app.state.cv.settings.tracker.auto_multi_ply = False  # exercise the review path
    b0, b1, b2 = boards("e2e4", "e7e5")
    path = write_video(tmp_path / "two.avi", [(1.5, b0, None), (1.0, b0, ["e4", "e5"]), (3.0, b2, None)])
    v = upload_video(client, path)
    job = start(client, v["id"]).json()
    paused = wait_job(client, job["id"], {"paused_for_review"})
    client.post(f"/api/jobs/{job['id']}/review", json={"event_id": paused["pending_review_event_id"], "action": "stop"})
    done = wait_job(client, job["id"], {"cancelled", "completed", "failed"})
    assert done["status"] == "cancelled"
    assert client.get(f"/api/games/{done['game_id']}").json()["moves"] == []
