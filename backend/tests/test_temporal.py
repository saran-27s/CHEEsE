"""Tracker tests on synthetic frame sequences (test fixtures, not real video)."""

import chess
import pytest

from chessvision.core.temporal import FrameEvidence, TemporalTracker, TrackerConfig, TrackerState
from fixtures import after, fixture_observation

START = chess.STARTING_FEN


def run(segments, fps, config=None, board=None):
    """segments: list of (duration_seconds, observation | None, occluded)."""
    tracker = TemporalTracker(board or chess.Board(), config or TrackerConfig())
    events = []
    t = 0.0
    dt = 1.0 / fps
    for duration, obs, occ in segments:
        end = t + duration
        while t < end - 1e-9:
            events += tracker.step(FrameEvidence(t=round(t, 6), observation=obs, occluded=occ))
            t += dt
    return tracker, events


def moves_of(events):
    return [e.uci for e in events if e.kind == "move"]


def test_hand_occlusion_then_move_is_committed_once():
    s0 = fixture_observation(START)
    s1 = fixture_observation(after(START, "e2e4"))
    tracker, events = run([(2, s0, False), (1.5, s1, True), (2, s1, False)], fps=10)
    assert moves_of(events) == ["e2e4"]
    assert tracker.state == TrackerState.STABLE
    assert tracker.board.fen() == after(START, "e2e4").fen()
    move = [e for e in events if e.kind == "move"][0]
    assert move.t_settled == pytest.approx(3.5, abs=0.11)


@pytest.mark.parametrize("fps", [3, 5, 10, 30, 60])
def test_frame_rate_independent(fps):
    s0 = fixture_observation(START)
    s1 = fixture_observation(after(START, "e2e4"))
    s2 = fixture_observation(after(START, "e2e4", "e7e5"))
    segs = [(2, s0, False), (1, s1, True), (2, s1, False), (1, s2, True), (2, s2, False)]
    tracker, events = run(segs, fps=fps)
    assert moves_of(events) == ["e2e4", "e7e5"]
    settled = [e.t_settled for e in events if e.kind == "move"]
    assert settled[0] == pytest.approx(3.0, abs=1.0 / fps + 1e-6)
    assert settled[1] == pytest.approx(6.0, abs=1.0 / fps + 1e-6)


def test_short_flicker_does_not_commit():
    s0 = fixture_observation(START)
    s1 = fixture_observation(after(START, "e2e4"))
    _, events = run([(2, s0, False), (0.4, s1, False), (2, s0, False)], fps=10)
    assert moves_of(events) == []


def test_occlusion_preserves_position_and_no_capture():
    s0 = fixture_observation(START)
    hidden = fixture_observation(START, occluded={"d8", "e8", "d7", "e7"})
    tracker, events = run([(1, s0, False), (3, hidden, False), (2, s0, False)], fps=10)
    assert moves_of(events) == []
    assert tracker.board.fen() == chess.STARTING_FEN


def test_missing_detections_are_dropouts_not_captures():
    s0 = fixture_observation(START)
    blank = fixture_observation("8/8/8/8/8/8/8/8 w - - 0 1")  # detector saw nothing
    tracker, events = run([(1, s0, False), (3, blank, False), (2, s0, False)], fps=10)
    assert events == [] or all(e.kind != "move" for e in events)
    assert tracker.board.fen() == chess.STARTING_FEN
    assert tracker.state == TrackerState.STABLE


def test_none_observation_frames_are_dropouts():
    s0 = fixture_observation(START)
    tracker, events = run([(1, s0, False), (2, None, False), (2, s0, False)], fps=10)
    assert moves_of(events) == [] and tracker.board.fen() == chess.STARTING_FEN


def test_two_moves_during_one_occlusion_needs_review():
    s0 = fixture_observation(START)
    s2 = fixture_observation(after(START, "e2e4", "e7e5"))
    tracker, events = run([(1, s0, False), (2, s2, True), (2, s2, False), (2, s2, False)], fps=10,
                          config=TrackerConfig(auto_multi_ply=False))
    reviews = [e for e in events if e.kind == "review"]
    assert len(reviews) == 1 and reviews[0].result.status == "invalid"
    assert tracker.state == TrackerState.NEEDS_REVIEW
    assert tracker.board.fen() == chess.STARTING_FEN  # nothing fabricated


def test_review_resolution_with_move_and_continue():
    s0 = fixture_observation(START)
    s1 = fixture_observation(after(START, "g1f3"), occluded={"f3"})
    tracker, events = run([(1, s0, False), (2, s1, False)], fps=10)
    assert tracker.state == TrackerState.NEEDS_REVIEW
    tracker.resolve_review("g1f3")
    assert tracker.state == TrackerState.STABLE
    assert tracker.board.fen() == after(START, "g1f3").fen()
    with pytest.raises(RuntimeError):
        tracker.resolve_review(None)


def test_review_dismissal_ignores_same_view():
    s0 = fixture_observation(START)
    noisy = fixture_observation(START, override={"d4": "Q"})  # persistent misdetection
    tracker, events = run([(1, s0, False), (2, noisy, False)], fps=10)
    assert tracker.state == TrackerState.NEEDS_REVIEW
    tracker.resolve_review(None)
    later = []
    t = 10.0
    for _ in range(40):
        later += tracker.step(FrameEvidence(t=t, observation=noisy))
        t += 0.1
    assert later == [] and tracker.state == TrackerState.STABLE


def test_illegal_review_move_rejected():
    s0 = fixture_observation(START)
    s2 = fixture_observation(after(START, "e2e4", "e7e5"))
    tracker, _ = run([(1, s0, False), (2, s2, False)], fps=10, config=TrackerConfig(auto_multi_ply=False))
    with pytest.raises(ValueError):
        tracker.resolve_review("e7e5")


def test_castling_en_passant_promotion_sequence():
    fen = "r3k3/pppp1p1P/8/4pP2/8/8/PPPPP1P1/R3K2R w KQq e6 0 1"
    seq = ["f5e6", "e8c8", "h7h8q"]
    boards = [fen]
    b = chess.Board(fen)
    for u in seq:
        b.push_uci(u)
        boards.append(b.fen())
    segs = []
    for i, f in enumerate(boards):
        if i:
            segs.append((1, fixture_observation(f), True))
        segs.append((2, fixture_observation(f), False))
    tracker, events = run(segs, fps=10, board=chess.Board(fen))
    assert moves_of(events) == seq
    descs = [e.result.best.description for e in events if e.kind == "move"]
    assert descs[0].is_en_passant and descs[1].is_castling and descs[2].promotion == "queen"


def test_two_moves_without_stable_view_recorded_as_sequence():
    s0 = fixture_observation(START)
    s2 = fixture_observation(after(START, "e2e4", "e7e5"))
    tracker, events = run([(1, s0, False), (2, s2, True), (2, s2, False)], fps=10)
    assert moves_of(events) == ["e2e4", "e7e5"]
    assert all(e.sequence is not None for e in events if e.kind == "move")
    assert tracker.state == TrackerState.STABLE


def test_end_of_stream_change_raises_review_not_move():
    s0 = fixture_observation(START)
    s1 = fixture_observation(after(START, "g1f3"))
    tracker, events = run([(1, s0, False), (1, s1, True), (0.3, s1, False)], fps=10)  # too short to be stable
    assert moves_of(events) == []
    final = tracker.finish()
    assert [e.kind for e in final] == ["review"] and final[0].end_of_stream
    tracker.resolve_review(["g1f3"])
    assert tracker.board.fen() == after(START, "g1f3").fen()


def test_persistent_misread_is_learned_and_ignored():
    s0 = fixture_observation(START, override={"d4": "Q"})  # detector always sees a phantom piece on d4
    s1 = fixture_observation(after(START, "e2e4"), override={"d4": "Q"})
    tracker, _ = run([(2, s0, False)], fps=10)
    assert tracker.state == TrackerState.NEEDS_REVIEW
    tracker.resolve_review(None)
    tracker.misreads = {chess.D4: "Q"}  # as learned after a confirmation
    events = []
    t = 10.0
    for obs, occ, n in ((s1, True, 10), (s1, False, 20)):
        for _ in range(n):
            events += tracker.step(FrameEvidence(t=t, observation=obs, occluded=occ))
            t += 0.1
    assert moves_of(events) == ["e2e4"]
