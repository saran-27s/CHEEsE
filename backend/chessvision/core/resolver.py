"""Resolve an observation against the legal successors of a confirmed position.

Candidates are "no move" plus every legal move (each promotion piece is a separate candidate).
Legality alone never confirms a move: the best candidate must also agree with every confidently
observed square, have all its changed squares confidently observed, and beat the runner-up by
a clear margin.
"""

from __future__ import annotations

import math
from typing import Literal

import chess
from pydantic import BaseModel

from .notation import MoveDescription, describe_move
from .observation import EMPTY, BoardObservation, log_prob

ResolveStatus = Literal["no_move", "confirmed", "ambiguous", "invalid"]


class Candidate(BaseModel):
    uci: str | None  # None = no move
    score: float
    margin_to_best: float
    changed_squares: list[str]
    mismatched_squares: list[str]
    unobserved_changed_squares: list[str]
    low_confidence_changed_squares: list[str]
    description: MoveDescription | None


class ResolveResult(BaseModel):
    status: ResolveStatus
    reason: str
    best: Candidate | None
    candidates: list[Candidate]
    margin: float | None
    before_fen: str
    after_fen: str | None
    observed_squares: int
    occluded_squares: int


def _expected(board: chess.Board) -> list[str]:
    out = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        out.append(p.symbol() if p else EMPTY)
    return out


def _cls(label: str) -> str:
    """Colour class of a label: empty, 'W' (white piece) or 'B' (black piece)."""
    if label == EMPTY:
        return EMPTY
    return "W" if label.isupper() else "B"


def _class_probs(sq_obs) -> dict[str, float]:
    out = {EMPTY: 0.0, "W": 0.0, "B": 0.0}
    for label, p in (sq_obs.probs or {}).items():
        out[_cls(label)] += p
    return out


def observed_key(sq_obs, colour: bool, uncertain_threshold: float) -> str | None:
    """Confident observed label of a square (colour class or piece label), else None."""
    if sq_obs.occluded or sq_obs.probs is None:
        return None
    if colour:
        p = _class_probs(sq_obs)
        k = max(p, key=p.get)
        return k if p[k] >= uncertain_threshold else None
    return sq_obs.top if sq_obs.top_prob >= uncertain_threshold else None


def _ignored(observation: BoardObservation, sq: int, ignore: dict[int, str], colour: bool, th: float) -> bool:
    """True when the square still shows its known, persistent misreading (so it carries no evidence)."""
    want = ignore.get(sq)
    return want is not None and observed_key(observation.squares[sq], colour, th) == want


def resolve(
    board: chess.Board,
    observation: BoardObservation,
    *,
    uncertain_threshold: float = 0.6,
    margin_threshold: float = 3.0,
    max_reported: int = 8,
    match: Literal["auto", "piece", "colour"] = "auto",
    unchanged_mismatch_tolerance: int | None = None,
    ignore: dict[int, str] | None = None,
) -> ResolveResult:
    """Score "no move" and every legal move against the observation.

    ``match`` selects how squares are compared:

    * ``piece``: exact piece labels (used for manual entry and fixtures, which are exact).
    * ``colour``: empty / white / black only. Detector piece-type errors (for example king vs
      queen) are common on unseen boards, while presence and colour are read reliably. Piece
      type then only separates candidates that share the same occupancy change (promotions).
    * ``auto``: ``colour`` for model observations, ``piece`` otherwise.

    ``unchanged_mismatch_tolerance`` is how many confident disagreements on squares the best
    candidate does not change are treated as detector misreads instead of evidence of a second
    move. A second move changes at least two squares, so one isolated disagreement is far more
    likely a misread. Default: 1 in colour mode, 0 in piece mode. Disagreement on a square the
    candidate changes is never tolerated.
    """
    if not board.is_valid():
        raise ValueError("cannot resolve from an invalid position")
    mode = match if match != "auto" else ("colour" if observation.source == "model" else "piece")
    colour = mode == "colour"
    tol = unchanged_mismatch_tolerance if unchanged_mismatch_tolerance is not None else (1 if colour else 0)
    before = _expected(board)
    known = [sq for sq in chess.SQUARES if not observation.squares[sq].occluded
             and observation.squares[sq].probs is not None]
    known_set = set(known)
    cprobs = {sq: _class_probs(observation.squares[sq]) for sq in known} if colour else {}
    if ignore:
        ignored_set = {sq for sq in known if _ignored(observation, sq, ignore, colour, uncertain_threshold)}
        known = [sq for sq in known if sq not in ignored_set]
        known_set = set(known)
    else:
        ignored_set = set()
    ignored_set |= {sq for sq in chess.SQUARES if "known_misread" in observation.squares[sq].reasons}

    def key(label: str) -> str:
        return _cls(label) if colour else label

    def top_of(sq: int) -> tuple[str, float]:
        if colour:
            p = cprobs[sq]
            k = max(p, key=p.get)
            return k, p[k]
        o = observation.squares[sq]
        return o.top, o.top_prob  # type: ignore[return-value]

    def logp(sq: int, label: str) -> float:
        if colour:
            return math.log(max(cprobs[sq][_cls(label)], 1e-6))
        return log_prob(observation.squares[sq], label)

    # (primary score, piece-level tiebreak score, move, expected labels)
    scored: list[tuple[float, float, chess.Move | None, list[str]]] = []
    moves: list[chess.Move | None] = [None, *board.legal_moves]
    for move in moves:
        if move is None:
            expected = before
        else:
            b = board.copy(stack=False)
            b.push(move)
            expected = _expected(b)
        primary = sum(logp(sq, expected[sq]) for sq in known)
        piece = sum(log_prob(observation.squares[sq], expected[sq]) for sq in known) if colour else primary
        scored.append((primary, piece, move, expected))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_score = scored[0][0]

    def build(score: float, piece_score: float, move: chess.Move | None, expected: list[str]) -> Candidate:
        changed = [sq for sq in chess.SQUARES if expected[sq] != before[sq]]
        mismatched = []
        for sq in known:
            k, p = top_of(sq)
            if k != key(expected[sq]) and p >= uncertain_threshold:
                mismatched.append(chess.square_name(sq))
        unobserved = [chess.square_name(sq) for sq in changed if sq not in known_set and sq not in ignored_set]
        low = []
        for sq in changed:
            if sq not in known_set:
                continue
            k, p = top_of(sq)
            if k != key(expected[sq]) or p < uncertain_threshold:
                low.append(chess.square_name(sq))
        return Candidate(
            uci=move.uci() if move else None, score=round(score, 4),
            margin_to_best=round(best_score - score, 4),
            changed_squares=[chess.square_name(sq) for sq in changed],
            mismatched_squares=mismatched, unobserved_changed_squares=unobserved,
            low_confidence_changed_squares=low,
            description=describe_move(board, move) if move else None,
        )

    top = [build(*t) for t in scored[:max_reported]]
    best = top[0]
    # Margin: best vs the best candidate with a different occupancy change, and (colour mode)
    # best vs a same-occupancy sibling by piece score (e.g. promotion to Q vs R).
    best_key = tuple(key(x) for x in scored[0][3])
    margin: float | None = None
    for primary, piece, _m, exp in scored[1:]:
        if tuple(key(x) for x in exp) != best_key:
            margin = scored[0][0] - primary
            break
    for primary, piece, _m, exp in scored[1:]:
        if tuple(key(x) for x in exp) == best_key:
            sib = scored[0][1] - piece
            margin = sib if margin is None else min(margin, sib)
            break
    occluded = 64 - len(known)

    def result(status: ResolveStatus, reason: str) -> ResolveResult:
        after_fen = None
        if status == "confirmed":
            after_fen = best.description.fen_after if best.description else None
        elif status == "no_move":
            after_fen = board.fen()
        return ResolveResult(status=status, reason=reason, best=best, candidates=top,
                             margin=round(margin, 4) if margin is not None else None,
                             before_fen=board.fen(), after_fen=after_fen,
                             observed_squares=len(known), occluded_squares=occluded)

    if not known:
        return result("ambiguous", "no squares observed")
    changed_set = set(best.changed_squares)
    mm_changed = [s for s in best.mismatched_squares if s in changed_set]
    mm_unchanged = [s for s in best.mismatched_squares if s not in changed_set]
    if mm_changed or len(mm_unchanged) > tol:
        return result("invalid", "observation contradicts every single-move explanation on "
                      + ", ".join(best.mismatched_squares)
                      + " (possible multiple moves or recognition error)")
    if best.unobserved_changed_squares:
        return result("ambiguous", "squares changed by the best candidate are not observed: "
                      + ", ".join(best.unobserved_changed_squares))
    if best.low_confidence_changed_squares:
        return result("ambiguous", "squares changed by the best candidate are not confidently observed: "
                      + ", ".join(best.low_confidence_changed_squares))
    if margin is not None and margin < margin_threshold:
        runner = top[1]
        return result("ambiguous", f"best candidate beats {runner.uci or 'no move'} by only {margin:.2f} "
                      f"(threshold {margin_threshold})")
    if best.uci is None:
        return result("no_move", "observation matches the current position")
    return result("confirmed", "single legal move explains the observation")


# ---- multi-move (sequence) resolution -------------------------------------------------------

SequenceStatus = Literal["confirmed", "ambiguous", "invalid"]


class SequenceCandidate(BaseModel):
    ucis: list[str]
    sans: list[str]
    score: float  # log-likelihood of the final observation minus the per-ply penalty
    margin_to_best: float
    mismatched_squares: list[str]
    changed_squares: list[str]
    orderings: int  # distinct legal move orders that reach the same final placement
    order_margin: float | None  # glimpse-evidence margin of the chosen order over the next one


class SequenceResult(BaseModel):
    status: SequenceStatus
    reason: str
    best: SequenceCandidate | None
    candidates: list[SequenceCandidate]
    margin: float | None


def _key_fn(colour: bool):
    return (lambda label: _cls(label)) if colour else (lambda label: label)


def _square_tables(observation: BoardObservation, colour: bool) -> dict[int, dict[str, float]]:
    """Per known square: label key -> probability (colour classes or piece labels)."""
    out: dict[int, dict[str, float]] = {}
    for sq in chess.SQUARES:
        s = observation.squares[sq]
        if s.occluded or s.probs is None:
            continue
        out[sq] = _class_probs(s) if colour else dict(s.probs)
    return out


def _score_keys(tables: dict[int, dict[str, float]], keys: list[str]) -> float:
    return sum(math.log(max(p.get(keys[sq], 0.0), 1e-6)) for sq, p in tables.items())


def resolve_sequence(
    board: chess.Board,
    observation: BoardObservation,
    *,
    max_plies: int = 3,
    uncertain_threshold: float = 0.6,
    margin_threshold: float = 3.0,
    ply_penalty: float = 2.0,
    match: Literal["auto", "piece", "colour"] = "auto",
    unchanged_mismatch_tolerance: int | None = None,
    glimpses: list[BoardObservation] | None = None,
    order_margin_threshold: float = 1.0,
    max_reported: int = 5,
    ignore: dict[int, str] | None = None,
) -> SequenceResult:
    """Explain an observation with a sequence of 1..``max_plies`` legal moves.

    Used when the board was never still long enough between moves (fast play). Sequences are
    grouped by the placement they produce; each extra ply costs ``ply_penalty`` nats so noise is
    not explained away with extra moves. The best placement must agree with every confidently
    observed square it changes, have all of them confidently observed, and beat the next
    placement by ``margin_threshold``.

    Several move orders can reach the same placement (for example W1 B W2 vs W2 B W1). The order
    is chosen with ``glimpses``: short, not-yet-stable views of the board seen between the moves.
    If they do not separate the orders by ``order_margin_threshold``, the result is ambiguous.
    """
    if not board.is_valid():
        raise ValueError("cannot resolve from an invalid position")
    mode = match if match != "auto" else ("colour" if observation.source == "model" else "piece")
    colour = mode == "colour"
    tol = unchanged_mismatch_tolerance if unchanged_mismatch_tolerance is not None else (1 if colour else 0)
    key = _key_fn(colour)
    tables = _square_tables(observation, colour)
    if ignore:
        tables = {sq: p for sq, p in tables.items()
                  if not _ignored(observation, sq, ignore, colour, uncertain_threshold)}
    before = _expected(board)

    # Enumerate sequences depth-first; group by full placement after the sequence.
    groups: dict[tuple, dict] = {}

    def visit(b: chess.Board, seq: list[chess.Move], positions: list[list[str]]) -> None:
        if seq:
            exp = positions[-1]
            placement = tuple(exp)
            primary = _score_keys(tables, [key(x) for x in exp]) - ply_penalty * (len(seq) - 1)
            g = groups.get(placement)
            if g is None or len(seq) < g["plies"]:
                groups[placement] = {"plies": len(seq), "score": primary, "expected": exp,
                                     "orders": [(list(seq), [list(p) for p in positions])]}
            elif len(seq) == g["plies"]:
                g["orders"].append((list(seq), [list(p) for p in positions]))
        if len(seq) >= max_plies:
            return
        for mv in list(b.legal_moves):
            b.push(mv)
            seq.append(mv)
            positions.append(_expected(b))
            visit(b, seq, positions)
            positions.pop()
            seq.pop()
            b.pop()

    visit(board.copy(), [], [before])
    # "no move" is a candidate too
    no_move_score = _score_keys(tables, [key(x) for x in before])
    ranked = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
    if not ranked:
        return SequenceResult(status="invalid", reason="no legal moves", best=None, candidates=[], margin=None)
    best_score = ranked[0]["score"]

    def glimpse_score(positions: list[list[str]]) -> float:
        """Best monotone assignment of glimpses to positions P0..Pk (Viterbi over a chain)."""
        if not glimpses:
            return 0.0
        k = len(positions)
        gt = [_square_tables(g, colour) for g in glimpses]
        pos_keys = [[key(x) for x in p] for p in positions]
        prev = [-math.inf] * k
        for i, t in enumerate(gt):
            row = [_score_keys(t, pos_keys[j]) for j in range(k)]
            cur = [-math.inf] * k
            best_so_far = -math.inf if i else 0.0
            for j in range(k):
                if i:
                    best_so_far = max(best_so_far, prev[j])
                cur[j] = best_so_far + row[j]
            prev = cur
        return max(prev)

    def build(g: dict) -> SequenceCandidate:
        exp = g["expected"]
        orders = g["orders"]
        order_margin = None
        if len(orders) > 1:
            scored = sorted(((glimpse_score(pos), seq) for seq, pos in orders), key=lambda x: x[0], reverse=True)
            order_margin = scored[0][0] - scored[1][0]
            seq = scored[0][1]
        else:
            seq = orders[0][0]
        sans, b = [], board.copy(stack=False)
        for mv in seq:
            sans.append(b.san(mv))
            b.push(mv)
        mismatched = []
        for sq, p in tables.items():
            top = max(p, key=p.get)
            if top != key(exp[sq]) and p[top] >= uncertain_threshold:
                mismatched.append(chess.square_name(sq))
        changed = [chess.square_name(sq) for sq in chess.SQUARES if exp[sq] != before[sq]]
        return SequenceCandidate(ucis=[m.uci() for m in seq], sans=sans, score=round(g["score"], 4),
                                 margin_to_best=round(best_score - g["score"], 4),
                                 mismatched_squares=mismatched, changed_squares=changed,
                                 orderings=len(orders),
                                 order_margin=round(order_margin, 4) if order_margin is not None else None)

    top = [build(g) for g in ranked[:max_reported]]
    best = top[0]
    best_keys = tuple(key(x) for x in ranked[0]["expected"])
    margin = best_score - no_move_score
    for g in ranked[1:]:
        if tuple(key(x) for x in g["expected"]) != best_keys:
            margin = min(margin, best_score - g["score"])
            break
    for g in ranked[1:]:
        if tuple(key(x) for x in g["expected"]) == best_keys:  # same colours, other piece types
            if colour:
                pt = _square_tables(observation, False)
                d = _score_keys(pt, ranked[0]["expected"]) - _score_keys(pt, g["expected"])
                margin = min(margin, d)
            break

    def out(status: SequenceStatus, reason: str) -> SequenceResult:
        return SequenceResult(status=status, reason=reason, best=best, candidates=top, margin=round(margin, 4))

    changed_set = set(best.changed_squares)
    mm_changed = [s for s in best.mismatched_squares if s in changed_set]
    mm_unchanged = [s for s in best.mismatched_squares if s not in changed_set]
    if mm_changed or len(mm_unchanged) > tol:
        return out("invalid", "no sequence of up to " f"{max_plies} moves explains the board (mismatch on "
                   + ", ".join(best.mismatched_squares) + ")")
    for name in best.changed_squares:
        sq = chess.parse_square(name)
        p = tables.get(sq)
        if p is None:
            if "known_misread" in observation.squares[sq].reasons or (
                    ignore and _ignored(observation, sq, ignore, colour, uncertain_threshold)):
                continue  # known misread square: no evidence either way, the other squares decide
            return out("ambiguous", f"square {name} changed by the best sequence is not observed")
        if p.get(key(ranked[0]["expected"][sq]), 0.0) < uncertain_threshold:
            return out("ambiguous", f"square {name} changed by the best sequence is not confidently observed")
    if margin < margin_threshold:
        return out("ambiguous", f"best sequence {' '.join(best.sans)} beats the next explanation by only "
                   f"{margin:.2f} (threshold {margin_threshold})")
    if best.orderings > 1 and (best.order_margin is None or best.order_margin < order_margin_threshold):
        return out("ambiguous", f"{best.orderings} move orders reach the same position and the frames between "
                   "moves do not show which came first")
    return out("confirmed", f"{len(best.ucis)} legal move(s) explain the board")
