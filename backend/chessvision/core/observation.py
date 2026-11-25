"""Per-square observations built from detections or from manual entry.

An observation is evidence. It never changes the confirmed game state by itself.
"""

from __future__ import annotations

import math
from typing import Literal

import chess
from pydantic import BaseModel, Field

from .coords import edge_distance_fraction, square_at, square_center, square_name

PIECE_LABELS = ("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
EMPTY = "empty"
HAND = "hand"
LABELS = (EMPTY, *PIECE_LABELS)

ObservationSource = Literal["model", "manual", "test_fixture"]


class Detection(BaseModel):
    label: str  # one of PIECE_LABELS or "hand"
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in canonical warped pixels


class AnchorSpec(BaseModel):
    """Where a detection box touches the board. Model-specific (see design section 4)."""

    mode: Literal["fraction", "toward_camera"] = "toward_camera"
    fx: float = 0.5
    fy: float = 0.5
    strength: float = 0.5


class SquareObservation(BaseModel):
    square: str
    probs: dict[str, float] | None  # None when occluded/unknown
    top: str | None
    top_prob: float
    occluded: bool = False
    uncertain: bool = False
    reasons: list[str] = Field(default_factory=list)
    detections: int = 0


class BoardObservation(BaseModel):
    source: ObservationSource
    squares: list[SquareObservation]  # index == python-chess square
    hand_detected: bool = False
    frame_occluded: bool = False
    warnings: list[str] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)

    def get(self, square: int) -> SquareObservation:
        return self.squares[square]

    def placement(self) -> dict[str, str]:
        """Top label for every non-empty, non-occluded square (square name -> FEN symbol)."""
        return {s.square: s.top for s in self.squares if s.top and s.top != EMPTY and not s.occluded}

    def piece_count(self) -> int:
        return sum(1 for s in self.squares if s.top and s.top != EMPTY and not s.occluded)

    def signature(self) -> tuple[str | None, ...]:
        return tuple(None if s.occluded else s.top for s in self.squares)

    def uncertain_squares(self) -> list[str]:
        return [s.square for s in self.squares if s.uncertain]


def anchor_point(det: Detection, spec: AnchorSpec, camera_dir: tuple[float, float] | None) -> tuple[float, float]:
    x1, y1, x2, y2 = det.box
    w, h = x2 - x1, y2 - y1
    if spec.mode == "fraction":
        return x1 + spec.fx * w, y1 + spec.fy * h
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if camera_dir is None:
        return cx, cy
    dx, dy = camera_dir
    return cx + dx * spec.strength * w / 2.0, cy + dy * spec.strength * h / 2.0


def _distribution(dets: list[Detection], empty_prior: float) -> dict[str, float]:
    if not dets:
        rest = (1.0 - empty_prior) / len(PIECE_LABELS)
        probs = {label: rest for label in PIECE_LABELS}
        probs[EMPTY] = empty_prior
        return probs
    scores: dict[str, float] = {}
    for d in dets:
        scores[d.label] = scores.get(d.label, 0.0) + max(d.conf, 1e-6)
    mass = min(0.98, max(d.conf for d in dets))
    total = sum(scores.values())
    probs = {label: 0.0 for label in LABELS}
    for label, s in scores.items():
        probs[label] = mass * s / total
    others = [label for label in PIECE_LABELS if label not in scores]
    leftover = 1.0 - mass
    probs[EMPTY] += leftover / 2.0
    if others:
        for label in others:
            probs[label] += (leftover / 2.0) / len(others)
    else:
        probs[EMPTY] += leftover / 2.0
    return probs


def build_observation(
    detections: list[Detection],
    size: int,
    *,
    anchor: AnchorSpec,
    camera_dir: tuple[float, float] | None,
    empty_prior: float = 0.92,
    uncertain_threshold: float = 0.6,
    edge_margin: float = 0.12,
    hand_dilation: float = 0.10,
    frame_occluded: bool = False,
    source: ObservationSource = "model",
) -> BoardObservation:
    per_square: dict[int, list[Detection]] = {}
    near_edge: set[int] = set()
    warnings: list[str] = []
    hands = [d for d in detections if d.label == HAND]
    for det in detections:
        if det.label == HAND:
            continue
        if det.label not in PIECE_LABELS:
            warnings.append(f"ignored detection with unknown label {det.label!r}")
            continue
        ax, ay = anchor_point(det, anchor, camera_dir)
        sq = square_at(ax, ay, size)
        if sq is None:
            warnings.append(f"{det.label} detection anchored outside the board at ({ax:.0f}, {ay:.0f}) ignored")
            continue
        per_square.setdefault(sq, []).append(det)
        if edge_distance_fraction(ax, ay, sq, size) < edge_margin:
            near_edge.add(sq)

    def occluded_by_hand(sq: int) -> bool:
        cx, cy = square_center(sq, size)
        for h in hands:
            x1, y1, x2, y2 = h.box
            dx, dy = (x2 - x1) * hand_dilation, (y2 - y1) * hand_dilation
            if x1 - dx <= cx <= x2 + dx and y1 - dy <= cy <= y2 + dy:
                return True
        return False

    squares: list[SquareObservation] = []
    for sq in chess.SQUARES:
        name = square_name(sq)
        if frame_occluded or occluded_by_hand(sq):
            squares.append(SquareObservation(square=name, probs=None, top=None, top_prob=0.0,
                                             occluded=True, uncertain=True, reasons=["occluded"]))
            continue
        dets = per_square.get(sq, [])
        probs = _distribution(dets, empty_prior)
        top = max(probs, key=probs.get)
        reasons = []
        if probs[top] < uncertain_threshold:
            reasons.append("low_confidence")
        if len(dets) > 1:
            reasons.append("multiple_detections")
        if sq in near_edge:
            reasons.append("near_edge")
        squares.append(SquareObservation(square=name, probs=probs, top=top, top_prob=probs[top],
                                         uncertain=bool(reasons), reasons=reasons, detections=len(dets)))
    return BoardObservation(source=source, squares=squares, hand_detected=bool(hands),
                            frame_occluded=frame_occluded, warnings=warnings, detections=detections)


def manual_observation(placement: dict[str, str], source: ObservationSource = "manual",
                       confidence: float = 0.999, unknown: set[str] | None = None) -> BoardObservation:
    """Observation from a user-entered placement (square name -> FEN symbol). Squares not listed are empty.

    ``unknown`` squares are marked occluded (the user said they cannot tell).
    """
    unknown = unknown or set()
    squares: list[SquareObservation] = []
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    for sq in chess.SQUARES:
        name = square_name(sq)
        if name in unknown:
            squares.append(SquareObservation(square=name, probs=None, top=None, top_prob=0.0,
                                             occluded=True, uncertain=True, reasons=["unknown"]))
            continue
        label = placement.get(name, EMPTY)
        if label not in LABELS:
            raise ValueError(f"invalid label {label!r} on {name}")
        probs = {lbl: rest for lbl in LABELS}
        probs[label] = confidence
        squares.append(SquareObservation(square=name, probs=probs, top=label, top_prob=confidence))
    return BoardObservation(source=source, squares=squares)


def aggregate(observations: list[BoardObservation], uncertain_threshold: float = 0.6) -> BoardObservation:
    """Average several observations of the same (static) board, e.g. a stable video window."""
    if not observations:
        raise ValueError("nothing to aggregate")
    n = len(observations)
    squares: list[SquareObservation] = []
    for sq in chess.SQUARES:
        obs = [o.squares[sq] for o in observations]
        known = [o for o in obs if not o.occluded and o.probs is not None]
        name = square_name(sq)
        if len(known) * 2 <= n:
            squares.append(SquareObservation(square=name, probs=None, top=None, top_prob=0.0,
                                             occluded=True, uncertain=True, reasons=["occluded"]))
            continue
        probs = {lbl: sum(o.probs[lbl] for o in known) / len(known) for lbl in LABELS}  # type: ignore[index]
        top = max(probs, key=probs.get)
        reasons = sorted({r for o in known for r in o.reasons} - {"low_confidence"})
        if probs[top] < uncertain_threshold:
            reasons.append("low_confidence")
        squares.append(SquareObservation(square=name, probs=probs, top=top, top_prob=probs[top],
                                         uncertain=bool(reasons), reasons=reasons,
                                         detections=max(o.detections for o in known)))
    source = observations[0].source
    return BoardObservation(source=source, squares=squares,
                            hand_detected=any(o.hand_detected for o in observations),
                            warnings=sorted({w for o in observations for w in o.warnings}))


def log_prob(sq_obs: SquareObservation, label: str) -> float:
    if sq_obs.probs is None:
        return 0.0
    return math.log(max(sq_obs.probs.get(label, 0.0), 1e-6))
