from __future__ import annotations

import chess
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..core.game import MoveEntry, fen_at, replay
from ..core.metadata import PositionMetadata
from ..core.observation import BoardObservation, manual_observation
from ..core.resolver import resolve
from ..storage.db import now_iso
from . import games_service as gs
from .state import AppState, get_state, require

router = APIRouter(prefix="/api")


class CompareRequest(BaseModel):
    before: gs.StartSpec
    after_analysis_id: str


class MoveIn(BaseModel):
    uci: str
    provenance: str = "user"
    evidence: dict | None = None


class GameCreate(BaseModel):
    title: str | None = None
    start: gs.StartSpec
    moves: list[MoveIn] = Field(default_factory=list)


class MoveUpdate(BaseModel):
    uci: str


@router.post("/compare")
def compare(req: CompareRequest, st: AppState = Depends(get_state)):
    board, meta, desc = gs.resolve_start(st.db, req.before)
    after = require(st.db.get("analyses", req.after_analysis_id), "after analysis")
    if after.get("correction"):
        obs = manual_observation(after["correction"]["placement"])
        evidence_source = "user-corrected placement of the after image"
    else:
        obs = BoardObservation.model_validate(after["observation"])
        evidence_source = ("model observation of the after image" if after["mode"] == "model"
                           else "manual entry of the after image")
    r = st.settings.resolver
    result = resolve(board, obs, uncertain_threshold=r.uncertain_threshold, margin_threshold=r.margin_threshold,
                     max_reported=r.max_candidates_reported)
    return {
        **result.model_dump(),
        "before": {"fen": board.fen(), "source": desc, "metadata": meta.model_dump(), "assumptions": meta.assumptions()},
        "after": {"analysis_id": after["id"], "evidence_source": evidence_source, "evidence": obs.model_dump()},
        "note": "Only a single move (or no move) between the two positions can be confirmed. "
                "If several moves may have happened, the result is ambiguous or invalid.",
    }


@router.post("/games")
def create_game(req: GameCreate, st: AppState = Depends(get_state)):
    board, meta, desc = gs.resolve_start(st.db, req.start)
    entries = []
    for m in req.moves:
        if m.provenance not in ("user", "resolver_image"):
            raise HTTPException(422, "moves created through this endpoint must be 'user' or 'resolver_image'")
        ev = BoardObservation.model_validate(m.evidence) if m.evidence else None
        entries.append(MoveEntry(uci=m.uci, provenance=m.provenance, evidence=ev))
    check = replay(board.fen(), entries, gs.resolver_for(st.settings))
    if check.detached:
        raise HTTPException(422, {"message": "moves are not valid from the start position",
                                  "detached": [d.model_dump(include={"reason"}) | {"uci": d.entry.uci}
                                               for d in check.detached]})
    ts = now_iso()
    game = st.db.insert("games", {"title": req.title, "start_fen": board.fen(), "start_metadata_json": meta.model_dump(),
                                  "source": desc, "created_at": ts, "updated_at": ts})
    gs.apply_entries(st.db, st.settings, game, entries)
    return gs.game_view(st.db, st.settings, st.db.get("games", game["id"]))


@router.get("/games")
def list_games(st: AppState = Depends(get_state)):
    rows = st.db.query("SELECT g.*, (SELECT COUNT(*) FROM moves m WHERE m.game_id = g.id AND m.status='active') "
                       "AS move_count FROM games g ORDER BY g.updated_at DESC")
    return [{"id": r["id"], "title": r["title"], "source": r["source"], "start_fen": r["start_fen"],
             "move_count": r["move_count"], "updated_at": r["updated_at"]} for r in rows]


@router.get("/games/{game_id}")
def get_game(game_id: str, st: AppState = Depends(get_state)):
    return gs.game_view(st.db, st.settings, require(st.db.get("games", game_id), "game"))


@router.post("/games/{game_id}/moves")
def append_move(game_id: str, req: MoveUpdate, st: AppState = Depends(get_state)):
    game = require(st.db.get("games", game_id), "game")
    board = gs.final_board(st.db, game)
    try:
        move = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(422, f"malformed UCI {req.uci!r}")
    if move not in board.legal_moves:
        raise HTTPException(422, f"{req.uci} is not legal in {board.fen()}")
    gs.apply_entries(st.db, st.settings, game, [*gs.entries(st.db, game_id), MoveEntry(uci=req.uci, provenance="user")])
    return gs.game_view(st.db, st.settings, game)


def _locate(st: AppState, game_id: str, ply: int):
    game = require(st.db.get("games", game_id), "game")
    entries = gs.entries(st.db, game_id)
    if not 1 <= ply <= len(entries):
        raise HTTPException(404, f"ply {ply} not found")
    return game, entries


@router.put("/games/{game_id}/moves/{ply}")
def edit_move(game_id: str, ply: int, req: MoveUpdate, st: AppState = Depends(get_state)):
    game, entries = _locate(st, game_id, ply)
    board = chess.Board(fen_at(game["start_fen"], [e.uci for e in entries], ply - 1))
    try:
        move = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(422, f"malformed UCI {req.uci!r}")
    if move not in board.legal_moves:
        raise HTTPException(422, f"{req.uci} is not legal at ply {ply} ({board.fen()})")
    old = entries[ply - 1]
    entries[ply - 1] = MoveEntry(uci=req.uci, provenance="user", t_settled=old.t_settled, t_accepted=old.t_accepted,
                                 thumbnail=old.thumbnail, id=old.id)
    result = gs.apply_entries(st.db, st.settings, game, entries)
    return {**gs.game_view(st.db, st.settings, game),
            "revalidation": {"detached_now": [{"uci": d.entry.uci, "reason": d.reason} for d in result.detached]}}


@router.delete("/games/{game_id}/moves/{ply}")
def delete_move(game_id: str, ply: int, st: AppState = Depends(get_state)):
    game, entries = _locate(st, game_id, ply)
    removed = entries.pop(ply - 1)
    result = gs.apply_entries(st.db, st.settings, game, entries, removed_ids=[removed.id] if removed.id else [])
    return {**gs.game_view(st.db, st.settings, game),
            "revalidation": {"detached_now": [{"uci": d.entry.uci, "reason": d.reason} for d in result.detached]}}


@router.delete("/games/{game_id}/detached")
def clear_detached(game_id: str, st: AppState = Depends(get_state)):
    game = require(st.db.get("games", game_id), "game")
    st.db.execute("DELETE FROM moves WHERE game_id = ? AND status = 'detached'", (game_id,))
    return gs.game_view(st.db, st.settings, game)


@router.get("/games/{game_id}/export/pgn", response_class=PlainTextResponse)
def export_pgn(game_id: str, st: AppState = Depends(get_state)):
    game = require(st.db.get("games", game_id), "game")
    view = gs.game_view(st.db, st.settings, game)
    return PlainTextResponse(view["pgn"], media_type="application/x-chess-pgn",
                             headers={"Content-Disposition": f'attachment; filename="game-{game_id[:8]}.pgn"'})


@router.get("/games/{game_id}/export/fen")
def export_fen(game_id: str, ply: int | None = Query(None, ge=0), st: AppState = Depends(get_state)):
    game = require(st.db.get("games", game_id), "game")
    ucis = [e.uci for e in gs.entries(st.db, game_id)]
    if ply is None:
        ply = len(ucis)
    if ply > len(ucis):
        raise HTTPException(404, f"ply {ply} beyond game length {len(ucis)}")
    fen = fen_at(game["start_fen"], ucis, ply)
    start_meta = PositionMetadata.model_validate(game["start_metadata"])
    meta = start_meta if ply == 0 else gs.current_metadata(st.db, game) if ply == len(ucis) else None
    fields = dict(zip(("placement", "side_to_move", "castling", "en_passant", "halfmove", "fullmove"), fen.split()))
    if meta is None:
        provenance = {k: ("assumed_default" if start_meta.provenance.get(k) == "assumed_default" and k != "en_passant"
                          else "derived_from_game") for k in start_meta.provenance}
    else:
        provenance = meta.provenance
    assumptions = [f"{k} is based on an assumed start value" for k, v in provenance.items() if v == "assumed_default"]
    return {"fen": fen, "ply": ply, "fields": fields, "provenance": provenance, "assumptions": assumptions}


@router.get("/games/{game_id}/export/json")
def export_json(game_id: str, st: AppState = Depends(get_state)):
    game = require(st.db.get("games", game_id), "game")
    view = gs.game_view(st.db, st.settings, game)
    evidence = {m["id"]: m["evidence"] for m in st.db.active_moves(game_id) if m.get("evidence")}
    return {"report_type": "chess-vision game", **view,
            "move_evidence": evidence,
            "notes": ["Moves with provenance 'resolver_*' were confirmed by the resolver from observations; "
                      "'user' moves were entered or chosen by the user.",
                      "Video timestamps mark when the board settled after a move, not when the piece was touched."],
            "generated_at": now_iso()}
