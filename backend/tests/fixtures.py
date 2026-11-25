"""Synthetic observation fixtures for unit tests.

These are TEST FIXTURES, not real inference. They build observations directly from a known
board so the resolver and tracker logic can be tested deterministically.
"""

from __future__ import annotations

import chess

from chessvision.core.observation import EMPTY, LABELS, BoardObservation, SquareObservation


def fixture_observation(
    board: chess.Board | str,
    *,
    conf: float = 0.95,
    occluded: set[str] | None = None,
    low_conf: dict[str, float] | None = None,
    override: dict[str, str] | None = None,
) -> BoardObservation:
    """Observation that "sees" ``board`` with ``conf`` on every square.

    occluded:  squares reported as hidden (e.g. under a hand)
    low_conf:  square -> top probability to use instead of ``conf``
    override:  square -> label actually "seen" (simulates a misdetection or a different board)
    """
    if isinstance(board, str):
        board = chess.Board(board)
    occluded = occluded or set()
    low_conf = low_conf or {}
    override = override or {}
    squares = []
    for sq in chess.SQUARES:
        name = chess.square_name(sq)
        if name in occluded:
            squares.append(SquareObservation(square=name, probs=None, top=None, top_prob=0.0,
                                             occluded=True, uncertain=True, reasons=["occluded"]))
            continue
        piece = board.piece_at(sq)
        label = override.get(name, piece.symbol() if piece else EMPTY)
        p = low_conf.get(name, conf)
        rest = (1 - p) / (len(LABELS) - 1)
        probs = {lbl: rest for lbl in LABELS}
        probs[label] = p
        squares.append(SquareObservation(square=name, probs=probs, top=label, top_prob=p,
                                         uncertain=p < 0.6, reasons=["low_confidence"] if p < 0.6 else []))
    return BoardObservation(source="test_fixture", squares=squares)


def after(fen: str, *ucis: str) -> chess.Board:
    b = chess.Board(fen)
    for u in ucis:
        b.push_uci(u)
    return b
