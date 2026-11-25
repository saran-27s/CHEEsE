"""Game persistence on top of the pure core.game replay logic."""

from __future__ import annotations

from typing import Literal

import chess
from fastapi import HTTPException
from pydantic import BaseModel

from ..config import Settings
from ..core.game import MoveEntry, replay, to_pgn
from ..core.metadata import (
    PositionMetadata, assemble, board_from_trusted_fen, metadata_from_board, standard_start_metadata,
)
from ..core.observation import BoardObservation
from ..core.resolver import resolve
from ..storage.db import Database, now_iso
from .state import file_url


class StartSpec(BaseModel):
    kind: Literal["start", "fen", "analysis"]
    fen: str | None = None
    analysis_id: str | None = None


def resolve_start(db: Database, spec: StartSpec, *, require_side_to_move: bool = True) -> tuple[chess.Board, PositionMetadata, str]:
    """Return (board, metadata, description) for a trusted starting position or raise 422."""
    if spec.kind == "start":
        return chess.Board(), standard_start_metadata(), "standard starting position"
    if spec.kind == "fen":
        if not spec.fen:
            raise HTTPException(422, "fen is required")
        try:
            board, meta = board_from_trusted_fen(spec.fen.strip())
        except ValueError as exc:
            raise HTTPException(422, f"invalid FEN: {exc}")
        return board, meta, "user-supplied FEN"
    row = db.get("analyses", spec.analysis_id or "")
    if row is None:
        raise HTTPException(404, "analysis not found")
    corr = row.get("correction")
    if not corr:
        raise HTTPException(422, "confirm the before position first: review the board and save it (correction)")
    meta = PositionMetadata.model_validate(corr["metadata"])
    if require_side_to_move and meta.provenance.get("side_to_move") != "user":
        raise HTTPException(422, "set the side to move for this position; it cannot be read from an image")
    check = assemble(corr["placement"], meta)
    if not check.valid:
        raise HTTPException(422, {"message": "position is not legal", "issues": check.issues})
    return chess.Board(check.fen), meta, f"confirmed analysis {row['id']}"


def _entries(db: Database, game_id: str) -> list[MoveEntry]:
    out = []
    for m in db.active_moves(game_id):
        ev = BoardObservation.model_validate(m["evidence"]) if m.get("evidence") else None
        out.append(MoveEntry(uci=m["uci"], provenance=m["provenance"], evidence=ev, t_settled=m["t_settled"],
                             t_accepted=m["t_accepted"], thumbnail=m["thumbnail"], id=m["id"]))
    return out


def resolver_for(settings: Settings):
    r = settings.resolver
    return lambda board, obs: resolve(board, obs, uncertain_threshold=r.uncertain_threshold,
                                      margin_threshold=r.margin_threshold)


def apply_entries(db: Database, settings: Settings, game: dict, entries: list[MoveEntry],
                  removed_ids: list[str] | None = None):
    """Replay entries, then persist kept moves as active and the rest as detached."""
    result = replay(game["start_fen"], entries, resolver_for(settings))
    for rid in removed_ids or []:
        db.execute("DELETE FROM moves WHERE id = ?", (rid,))
    for m in result.moves:
        e = m.entry
        values = {"ply": m.ply, "uci": e.uci, "provenance": e.provenance, "status": "active",
                  "evidence_json": e.evidence.model_dump() if e.evidence else None, "detached_reason": None}
        if e.id:
            db.update("moves", e.id, values)
        else:
            db.insert("moves", {**values, "game_id": game["id"], "t_settled": e.t_settled,
                                "t_accepted": e.t_accepted, "thumbnail": e.thumbnail})
    for d in result.detached:
        e = d.entry
        values = {"status": "detached", "detached_reason": d.reason, "ply": None, "uci": e.uci,
                  "provenance": e.provenance}
        if e.id:
            db.update("moves", e.id, values)
        else:
            db.insert("moves", {**values, "game_id": game["id"],
                                "evidence_json": e.evidence.model_dump() if e.evidence else None})
    db.update("games", game["id"], {"updated_at": now_iso()})
    return result


def game_view(db: Database, settings: Settings, game: dict) -> dict:
    entries = _entries(db, game["id"])
    result = replay(game["start_fen"], entries, None)  # stored moves are already validated
    start_meta = PositionMetadata.model_validate(game["start_metadata"])
    moves = []
    for m in result.moves:
        moves.append({
            "ply": m.ply, "id": m.entry.id, "move_number": chess.Board(m.fen_before).fullmove_number,
            **m.description.model_dump(),
            "provenance": m.entry.provenance, "t_settled": m.entry.t_settled, "t_accepted": m.entry.t_accepted,
            "thumbnail_url": file_url("thumbnails", m.entry.thumbnail), "has_evidence": m.entry.evidence is not None,
            "fen_before": m.fen_before,
        })
    detached = [{"id": d["id"], "uci": d["uci"], "provenance": d["provenance"], "reason": d["detached_reason"],
                 "t_settled": d["t_settled"], "thumbnail_url": file_url("thumbnails", d["thumbnail"])}
                for d in db.detached_moves(game["id"])]
    return {
        "id": game["id"], "title": game["title"], "source": game["source"], "start_fen": game["start_fen"],
        "start_metadata": start_meta.model_dump(), "start_assumptions": start_meta.assumptions(),
        "moves": moves, "detached": detached, "final_fen": result.final_fen,
        "pgn": to_pgn(game["start_fen"], result.moves, {"Event": game["title"] or "chess-vision export"}, start_meta),
        "created_at": game["created_at"], "updated_at": game["updated_at"],
    }


def final_board(db: Database, game: dict) -> chess.Board:
    result = replay(game["start_fen"], _entries(db, game["id"]), None)
    return chess.Board(result.final_fen)


def current_metadata(db: Database, game: dict) -> PositionMetadata:
    board = final_board(db, game)
    start_meta = PositionMetadata.model_validate(game["start_metadata"])
    if not db.active_moves(game["id"]):
        return start_meta
    meta = metadata_from_board(board, "derived_from_game")
    # Fields that were assumptions at the start remain assumptions downstream.
    for f, src in start_meta.provenance.items():
        if src == "assumed_default" and f in ("castling", "en_passant", "halfmove", "fullmove", "side_to_move"):
            meta.provenance[f] = "assumed_default" if f != "en_passant" else "derived_from_game"
    return meta


def entries(db: Database, game_id: str) -> list[MoveEntry]:
    return _entries(db, game_id)
