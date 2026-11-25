"""Position metadata (side to move, castling, en passant, counters) with explicit provenance.

A single image only shows piece placement. Every other FEN field must come from the user,
a trusted FEN, the standard start, or the confirmed game. Anything else is an assumed default
and is labelled as such. Castling rights are never inferred from piece placement.
"""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, Field

Provenance = Literal["user", "trusted_fen", "standard_start", "derived_from_game", "assumed_default"]
FIELDS = ("side_to_move", "castling", "en_passant", "halfmove", "fullmove")

STANDARD_PLACEMENT_FEN = chess.STARTING_BOARD_FEN

_STATUS_MESSAGES = {
    chess.STATUS_NO_WHITE_KING: "no white king",
    chess.STATUS_NO_BLACK_KING: "no black king",
    chess.STATUS_TOO_MANY_KINGS: "too many kings",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "more than 8 white pawns",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "more than 8 black pawns",
    chess.STATUS_PAWNS_ON_BACKRANK: "pawn on the first or eighth rank",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "more than 16 white pieces",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "more than 16 black pieces",
    chess.STATUS_BAD_CASTLING_RIGHTS: "castling rights do not match king/rook placement",
    chess.STATUS_INVALID_EP_SQUARE: "en passant square is not possible in this position",
    chess.STATUS_OPPOSITE_CHECK: "the side not to move is in check",
    chess.STATUS_EMPTY: "board is empty",
    chess.STATUS_RACE_CHECK: "race check",
    chess.STATUS_RACE_OVER: "race over",
    chess.STATUS_RACE_MATERIAL: "race material",
    chess.STATUS_TOO_MANY_CHECKERS: "too many checkers",
    chess.STATUS_IMPOSSIBLE_CHECK: "impossible check configuration",
}

_CASTLING_REQUIREMENTS = {
    "K": ("e1", "K", "h1", "R"),
    "Q": ("e1", "K", "a1", "R"),
    "k": ("e8", "k", "h8", "r"),
    "q": ("e8", "k", "a8", "r"),
}


class PositionMetadata(BaseModel):
    side_to_move: Literal["w", "b"] = "w"
    castling: str = "-"
    en_passant: str = "-"
    halfmove: int = Field(0, ge=0)
    fullmove: int = Field(1, ge=1)
    provenance: dict[str, Provenance] = Field(default_factory=lambda: {f: "assumed_default" for f in FIELDS})

    def assumptions(self) -> list[str]:
        labels = {
            "side_to_move": f"Side to move assumed: {'White' if self.side_to_move == 'w' else 'Black'}",
            "castling": f"Castling rights assumed: {self.castling} (not inferred from piece placement)",
            "en_passant": f"En passant assumed: {self.en_passant}",
            "halfmove": f"Halfmove clock assumed: {self.halfmove}",
            "fullmove": f"Fullmove number assumed: {self.fullmove}",
        }
        return [labels[f] for f in FIELDS if self.provenance.get(f, "assumed_default") == "assumed_default"]


def default_metadata() -> PositionMetadata:
    return PositionMetadata()


def standard_start_metadata() -> PositionMetadata:
    return PositionMetadata(side_to_move="w", castling="KQkq", en_passant="-", halfmove=0, fullmove=1,
                            provenance={f: "standard_start" for f in FIELDS})


def metadata_from_board(board: chess.Board, source: Provenance) -> PositionMetadata:
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return PositionMetadata(
        side_to_move="w" if board.turn == chess.WHITE else "b",
        castling=board.castling_xfen() if board.castling_rights else "-",
        en_passant=ep,
        halfmove=board.halfmove_clock,
        fullmove=board.fullmove_number,
        provenance={f: source for f in FIELDS},
    )


def placement_to_board_fen(placement: dict[str, str]) -> str:
    board = chess.BaseBoard.empty()
    for name, symbol in placement.items():
        board.set_piece_at(chess.parse_square(name), chess.Piece.from_symbol(symbol))
    return board.board_fen()


def board_fen_to_placement(board_fen: str) -> dict[str, str]:
    board = chess.BaseBoard(board_fen)
    return {chess.square_name(sq): p.symbol() for sq, p in board.piece_map().items()}


def castling_errors(placement: dict[str, str], castling: str) -> list[str]:
    if castling in ("", "-"):
        return []
    errors = []
    for right in castling:
        if right not in _CASTLING_REQUIREMENTS:
            errors.append(f"unsupported castling flag {right!r} (standard chess only)")
            continue
        ksq, king, rsq, rook = _CASTLING_REQUIREMENTS[right]
        if placement.get(ksq) != king or placement.get(rsq) != rook:
            errors.append(f"castling right {right!r} needs {king} on {ksq} and {rook} on {rsq}")
    if len(set(castling)) != len(castling):
        errors.append("duplicate castling flags")
    return errors


class PositionCheck(BaseModel):
    fen: str
    valid: bool
    issues: list[str]
    assumptions: list[str]
    metadata: PositionMetadata


def assemble(placement: dict[str, str], metadata: PositionMetadata) -> PositionCheck:
    """Build a FEN from placement + metadata and report every problem without silently fixing it."""
    issues = castling_errors(placement, metadata.castling)
    board_fen = placement_to_board_fen(placement)
    fen = f"{board_fen} {metadata.side_to_move} {metadata.castling or '-'} {metadata.en_passant or '-'} " \
          f"{metadata.halfmove} {metadata.fullmove}"
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return PositionCheck(fen=fen, valid=False, issues=[*issues, f"invalid FEN: {exc}"],
                             assumptions=metadata.assumptions(), metadata=metadata)
    status = board.status()
    for flag, message in _STATUS_MESSAGES.items():
        if status & flag and message not in issues and not (flag == chess.STATUS_BAD_CASTLING_RIGHTS and issues):
            issues.append(message)
    return PositionCheck(fen=fen, valid=not issues, issues=issues,
                         assumptions=metadata.assumptions(), metadata=metadata)


def board_from_trusted_fen(fen: str) -> tuple[chess.Board, PositionMetadata]:
    board = chess.Board(fen)  # raises ValueError on malformed FEN
    if not board.is_valid():
        status = board.status()
        problems = [m for f, m in _STATUS_MESSAGES.items() if status & f]
        raise ValueError("illegal position: " + ", ".join(problems))
    return board, metadata_from_board(board, "trusted_fen")
