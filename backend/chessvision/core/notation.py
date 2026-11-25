"""Deterministic notation and English descriptions generated from python-chess moves."""

from __future__ import annotations

import chess
from pydantic import BaseModel

PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def _color(c: chess.Color) -> str:
    return "White" if c == chess.WHITE else "Black"


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


class MoveDescription(BaseModel):
    san: str
    uci: str
    from_square: str
    to_square: str
    coordinates: str
    english: str
    color: str
    piece: str
    is_capture: bool
    captured_piece: str | None
    is_en_passant: bool
    is_castling: bool
    castling_side: str | None
    promotion: str | None
    is_check: bool
    is_checkmate: bool
    fen_after: str


def describe_move(board: chess.Board, move: chess.Move) -> MoveDescription:
    """Describe ``move`` played from ``board`` (board is not modified). Raises ValueError if illegal."""
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move.uci()} in {board.fen()}")
    piece = board.piece_at(move.from_square)
    assert piece is not None
    mover = _color(piece.color)
    opponent = _color(not piece.color).lower()
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    san = board.san(move)
    is_ep = board.is_en_passant(move)
    is_castle = board.is_castling(move)
    captured_sq = None
    captured_type = None
    if board.is_capture(move):
        if is_ep:
            captured_sq = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        else:
            captured_sq = move.to_square
        cp = board.piece_at(captured_sq)
        captured_type = cp.piece_type if cp else chess.PAWN

    after = board.copy(stack=False)
    after.push(move)
    is_mate = after.is_checkmate()
    is_check = after.is_check()

    side = None
    if is_castle:
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(7 if side == "kingside" else 0, rank)
        rook_to = chess.square(5 if side == "kingside" else 3, rank)
        english = (f"{mover} castles {side}: king from {frm} to {to}, rook from "
                   f"{chess.square_name(rook_from)} to {chess.square_name(rook_to)}.")
    else:
        name = PIECE_NAMES[piece.piece_type]
        english = f"{mover} moves the {name} from {frm} to {to}"
        if captured_type is not None:
            cname = PIECE_NAMES[captured_type]
            if is_ep:
                english += f", capturing the {opponent} pawn on {chess.square_name(captured_sq)} en passant"
            else:
                english += f", capturing the {opponent} {cname}"
        if move.promotion:
            pname = PIECE_NAMES[move.promotion]
            joiner = ", and" if captured_type is not None else " and"
            english += f"{joiner} promotes it to {_article(pname)} {pname}"
        english += "."
    if is_mate:
        english += " This is checkmate."
    elif is_check:
        english += " This gives check."

    return MoveDescription(
        san=san, uci=move.uci(), from_square=frm, to_square=to, coordinates=f"{frm} → {to}",
        english=english, color=mover.lower(), piece=PIECE_NAMES[piece.piece_type],
        is_capture=captured_type is not None,
        captured_piece=PIECE_NAMES[captured_type] if captured_type is not None else None,
        is_en_passant=is_ep, is_castling=is_castle, castling_side=side,
        promotion=PIECE_NAMES[move.promotion] if move.promotion else None,
        is_check=is_check, is_checkmate=is_mate, fen_after=after.fen(),
    )
