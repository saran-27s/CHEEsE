import chess

from chessvision.core.game import MoveEntry, replay, to_pgn
from chessvision.core.resolver import resolve
from fixtures import after, fixture_observation

START = chess.STARTING_FEN


def test_no_move():
    r = resolve(chess.Board(), fixture_observation(START))
    assert r.status == "no_move" and r.best.uci is None


def test_confirmed_simple_move():
    r = resolve(chess.Board(), fixture_observation(after(START, "g1f3")))
    assert r.status == "confirmed"
    assert r.best.uci == "g1f3" and r.best.description.san == "Nf3"
    assert r.after_fen == after(START, "g1f3").fen()


def test_capture_confirmed():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    r = resolve(chess.Board(fen), fixture_observation(after(fen, "e4d5")))
    assert r.status == "confirmed" and r.best.uci == "e4d5"


def test_castling_confirmed_with_four_changed_squares():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    r = resolve(chess.Board(fen), fixture_observation(after(fen, "e1g1")))
    assert r.status == "confirmed" and r.best.uci == "e1g1"
    assert sorted(r.best.changed_squares) == ["e1", "f1", "g1", "h1"]


def test_en_passant_confirmed_with_three_changed_squares():
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    r = resolve(chess.Board(fen), fixture_observation(after(fen, "e5f6")))
    assert r.status == "confirmed" and r.best.uci == "e5f6"
    assert sorted(r.best.changed_squares) == ["e5", "f5", "f6"]


def test_each_promotion_choice_is_distinguished():
    fen = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"
    for p in "qrbn":
        r = resolve(chess.Board(fen), fixture_observation(after(fen, "e7e8" + p)))
        assert r.status == "confirmed" and r.best.uci == "e7e8" + p


def test_promotion_piece_unobserved_is_ambiguous():
    fen = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"
    obs = fixture_observation(after(fen, "e7e8q"), occluded={"e8"})
    r = resolve(chess.Board(fen), obs)
    assert r.status == "ambiguous"


def test_occluded_destination_is_ambiguous_not_capture():
    # Knight moved, but its destination square is hidden by a hand.
    obs = fixture_observation(after(START, "g1f3"), occluded={"f3"})
    r = resolve(chess.Board(), obs)
    assert r.status == "ambiguous"


def test_hidden_piece_is_not_a_capture():
    # Nothing moved; one black piece hidden. Must not produce a capture or a move.
    obs = fixture_observation(START, occluded={"d8"})
    r = resolve(chess.Board(), obs)
    assert r.status == "no_move"


def test_low_confidence_changed_square_is_ambiguous():
    obs = fixture_observation(after(START, "e2e4"), low_conf={"e4": 0.5})
    r = resolve(chess.Board(), obs)
    assert r.status == "ambiguous"


def test_two_moves_at_once_is_invalid():
    obs = fixture_observation(after(START, "e2e4", "e7e5"))
    r = resolve(chess.Board(), obs)
    assert r.status == "invalid"
    assert r.after_fen is None


def test_misdetection_is_invalid_not_forced():
    obs = fixture_observation(START, override={"d4": "Q"})
    r = resolve(chess.Board(), obs)
    assert r.status == "invalid"


def test_wrong_side_move_is_invalid():
    # Black-looking move while White is to move.
    board = chess.Board()
    obs = fixture_observation(after(START.replace(" w ", " b "), "e7e5"))
    assert resolve(board, obs).status == "invalid"


def test_replay_detaches_downstream_on_edit():
    entries = [MoveEntry(uci=u, provenance="user") for u in ("e2e4", "e7e5", "g1f3", "b8c6")]
    res = replay(START, entries)
    assert len(res.moves) == 4 and not res.detached
    # Change move 1 to d2d4: e7e5, g1f3, b8c6 remain legal so they are kept.
    edited = [MoveEntry(uci="d2d4", provenance="user"), *entries[1:]]
    res2 = replay(START, edited)
    assert [m.entry.uci for m in res2.moves] == ["d2d4", "e7e5", "g1f3", "b8c6"]
    # Change move 1 to g1h3: g1f3 becomes illegal -> it and everything after are detached.
    edited = [MoveEntry(uci="g1h3", provenance="user"), *entries[1:]]
    res3 = replay(START, edited)
    assert [m.entry.uci for m in res3.moves] == ["g1h3", "e7e5"]
    assert [d.entry.uci for d in res3.detached] == ["g1f3", "b8c6"]


def test_replay_re_resolves_evidence():
    ev = fixture_observation(after(START, "e2e4", "e7e5"))
    entries = [MoveEntry(uci="e2e4", provenance="user"),
               MoveEntry(uci="e7e5", provenance="resolver_video", evidence=ev)]
    assert len(replay(START, entries, resolve).moves) == 2
    # If move 1 is changed to d2d4, the stored evidence (pawn on e4) no longer supports e7e5.
    entries[0] = MoveEntry(uci="d2d4", provenance="user")
    res = replay(START, entries, resolve)
    assert [m.entry.uci for m in res.moves] == ["d2d4"]
    assert res.detached[0].entry.uci == "e7e5" and "evidence" in res.detached[0].reason


def test_pgn_standard_and_custom_start():
    entries = [MoveEntry(uci=u, provenance="user") for u in ("e2e4", "e7e5", "g1f3")]
    pgn = to_pgn(START, replay(START, entries).moves)
    assert "1. e4 e5 2. Nf3 *" in pgn and "[FEN" not in pgn
    fen = "4k3/8/8/8/8/8/8/R3K3 b - - 0 1"
    pgn2 = to_pgn(fen, replay(fen, [MoveEntry(uci="e8d7", provenance="user", t_settled=3.5)]).moves)
    assert '[SetUp "1"]' in pgn2 and f'[FEN "{fen}"]' in pgn2
    assert "1... Kd7" in pgn2 and "board settled at 3.50 s" in pgn2
