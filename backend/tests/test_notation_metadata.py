import chess
import pytest

from chessvision.core.metadata import (
    PositionMetadata, assemble, board_fen_to_placement, board_from_trusted_fen, castling_errors,
    default_metadata,
)
from chessvision.core.notation import describe_move


def d(fen, uci):
    return describe_move(chess.Board(fen), chess.Move.from_uci(uci))


def test_knight_move_full_output():
    m = d(chess.STARTING_FEN, "g1f3")
    assert m.san == "Nf3"
    assert m.uci == "g1f3"
    assert m.coordinates == "g1 → f3"
    assert m.english == "White moves the knight from g1 to f3."
    assert not (m.is_capture or m.is_castling or m.promotion or m.is_check)


def test_capture():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    m = d(fen, "e4d5")
    assert m.san == "exd5" and m.is_capture and m.captured_piece == "pawn"
    assert m.english == "White moves the pawn from e4 to d5, capturing the black pawn."


def test_castling_both_sides():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    k = d(fen, "e1g1")
    assert k.san == "O-O" and k.is_castling and k.castling_side == "kingside"
    assert k.english == "White castles kingside: king from e1 to g1, rook from h1 to f1."
    q = d(fen.replace(" w ", " b "), "e8c8")
    assert q.san == "O-O-O" and q.castling_side == "queenside"
    assert q.english == "Black castles queenside: king from e8 to c8, rook from a8 to d8."


def test_en_passant():
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    m = d(fen, "e5f6")
    assert m.is_en_passant and m.is_capture and m.san == "exf6"
    assert m.english == "White moves the pawn from e5 to f6, capturing the black pawn on f5 en passant."


@pytest.mark.parametrize("piece,word,san", [("q", "a queen", "e8=Q"), ("r", "a rook", "e8=R"),
                                             ("b", "a bishop", "e8=B"), ("n", "a knight", "e8=N")])
def test_all_promotions(piece, word, san):
    fen = "8/4P3/8/8/8/8/k7/4K3 w - - 0 1"
    m = d(fen, "e7e8" + piece)
    assert m.san.startswith(san)
    assert m.english.startswith(f"White moves the pawn from e7 to e8 and promotes it to {word}.")


def test_capture_promotion_and_check():
    fen = "3r4/4P3/8/8/8/8/k7/4K3 w - - 0 1"
    m = d(fen, "e7d8q")
    assert m.san == "exd8=Q"
    assert m.english == "White moves the pawn from e7 to d8, capturing the black rook, and promotes it to a queen."


def test_check_and_mate_flags():
    m = d("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2", "d8h4")
    assert m.is_checkmate and m.san == "Qh4#" and m.english.endswith("This is checkmate.")
    c = d("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8")
    assert c.is_check and not c.is_checkmate and c.english.endswith("This gives check.")


def test_illegal_move_rejected():
    with pytest.raises(ValueError):
        d(chess.STARTING_FEN, "e2e5")


def test_defaults_are_labelled_assumptions():
    placement = board_fen_to_placement(chess.STARTING_BOARD_FEN)
    check = assemble(placement, default_metadata())
    assert check.fen.endswith(" w - - 0 1")
    assert len(check.assumptions) == 5
    assert any("not inferred from piece placement" in a for a in check.assumptions)


def test_castling_never_inferred_and_impossible_claim_rejected():
    placement = board_fen_to_placement(chess.STARTING_BOARD_FEN)
    # Kings and rooks on home squares do not grant rights: default stays "-".
    assert assemble(placement, default_metadata()).fen.split()[2] == "-"
    del placement["h1"]
    meta = PositionMetadata(castling="KQkq", provenance={"castling": "user"})
    check = assemble(placement, meta)
    assert not check.valid
    assert any("'K'" in i for i in check.issues)
    assert castling_errors(placement, "Qkq") == []


def test_position_issues_reported():
    check = assemble({"e1": "K"}, default_metadata())
    assert not check.valid and "no black king" in check.issues


def test_trusted_fen():
    board, meta = board_from_trusted_fen(chess.STARTING_FEN)
    assert meta.provenance["castling"] == "trusted_fen" and meta.castling == "KQkq"
    with pytest.raises(ValueError):
        board_from_trusted_fen("8/8/8/8/8/8/8/8 w - - 0 1")
