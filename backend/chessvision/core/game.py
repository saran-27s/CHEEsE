"""Confirmed game record: start position + moves with provenance, replay and PGN export."""

from __future__ import annotations

from datetime import date
from typing import Callable, Literal

import chess
import chess.pgn
from pydantic import BaseModel

from .metadata import PositionMetadata
from .notation import MoveDescription, describe_move
from .observation import BoardObservation
from .resolver import ResolveResult

MoveProvenance = Literal["resolver_image", "resolver_video", "user"]


class MoveEntry(BaseModel):
    uci: str
    provenance: MoveProvenance
    evidence: BoardObservation | None = None
    t_settled: float | None = None
    t_accepted: float | None = None
    thumbnail: str | None = None
    id: str | None = None


class ReplayedMove(BaseModel):
    ply: int
    entry: MoveEntry
    description: MoveDescription
    fen_before: str


class Detached(BaseModel):
    entry: MoveEntry
    reason: str


class ReplayResult(BaseModel):
    moves: list[ReplayedMove]
    detached: list[Detached]
    final_fen: str


Resolver = Callable[[chess.Board, BoardObservation], ResolveResult]


def replay(start_fen: str, entries: list[MoveEntry], resolver: Resolver | None = None) -> ReplayResult:
    """Replay entries from ``start_fen``.

    Entries with evidence are re-resolved from the current position and kept only if the
    resolver confirms the same move. Entries without evidence are kept if legal. The first
    entry that fails, and every entry after it, are detached (never bridged or guessed).
    """
    board = chess.Board(start_fen)
    kept: list[ReplayedMove] = []
    detached: list[Detached] = []
    for i, entry in enumerate(entries):
        reason = None
        try:
            move = chess.Move.from_uci(entry.uci)
        except ValueError:
            move = None
            reason = f"malformed UCI {entry.uci!r}"
        if move is not None and move not in board.legal_moves:
            reason = f"{entry.uci} is not legal after ply {len(kept)}"
        if reason is None and entry.evidence is not None and resolver is not None and entry.provenance != "user":
            res = resolver(board, entry.evidence)
            if res.status != "confirmed" or res.best is None or res.best.uci != entry.uci:
                found = res.best.uci if res.status == "confirmed" and res.best else res.status
                reason = f"stored evidence no longer confirms {entry.uci} from this position (resolver: {found})"
        if reason is not None:
            detached.append(Detached(entry=entry, reason=reason))
            detached.extend(Detached(entry=e, reason="follows a detached move") for e in entries[i + 1:])
            break
        assert move is not None
        kept.append(ReplayedMove(ply=len(kept) + 1, entry=entry, description=describe_move(board, move),
                                 fen_before=board.fen()))
        board.push(move)
    return ReplayResult(moves=kept, detached=detached, final_fen=board.fen())


def fen_at(start_fen: str, ucis: list[str], ply: int) -> str:
    board = chess.Board(start_fen)
    for uci in ucis[:ply]:
        board.push_uci(uci)
    return board.fen()


def to_pgn(start_fen: str, moves: list[ReplayedMove], headers: dict[str, str] | None = None,
           start_metadata: PositionMetadata | None = None) -> str:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "chess-vision export"
    game.headers["Date"] = date.today().strftime("%Y.%m.%d")
    if start_fen != chess.STARTING_FEN:
        game.setup(board)
    for k, v in (headers or {}).items():
        game.headers[k] = v
    if start_metadata is not None:
        assumed = start_metadata.assumptions()
        if assumed:
            game.comment = "Start position assumptions: " + "; ".join(assumed)
    node: chess.pgn.GameNode = game
    for m in moves:
        node = node.add_variation(chess.Move.from_uci(m.entry.uci))
        if m.entry.t_settled is not None:
            node.comment = f"board settled at {m.entry.t_settled:.2f} s"
        board.push_uci(m.entry.uci)
    outcome = board.outcome()
    game.headers["Result"] = outcome.result() if outcome else "*"
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    return game.accept(exporter)
