"""Temporal state machine for move detection in a frame stream (recorded video now, live later).

STABLE -> OCCLUDED_OR_MOVING -> SETTLING -> VALIDATING -> STABLE, with NEEDS_REVIEW for
unresolved events. Stability is measured in media seconds plus a minimum frame count, so
results do not depend on the frame rate. The confirmed board only changes through a
``confirmed`` resolver result or an explicit ``resolve_review`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import chess

from .observation import EMPTY, BoardObservation, aggregate
from .resolver import ResolveResult, SequenceResult, resolve, resolve_sequence


class TrackerState(str, Enum):
    STABLE = "STABLE"
    OCCLUDED_OR_MOVING = "OCCLUDED_OR_MOVING"
    SETTLING = "SETTLING"
    VALIDATING = "VALIDATING"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class FrameEvidence:
    t: float  # media time in seconds
    observation: BoardObservation | None
    occluded: bool = False  # hand detected, or motion when no hand detector exists
    ref: Any = None  # opaque payload returned with events (e.g. the frame image for thumbnails)


@dataclass
class TrackerEvent:
    kind: str  # "move" | "review" | "no_move"
    t_settled: float
    t_accepted: float
    result: ResolveResult
    evidence: BoardObservation
    ref: Any = None
    uci: str | None = None
    sequence: SequenceResult | None = None  # multi-move explanation (review suggestion or auto-confirmed)
    sequence_index: int = 0
    sequence_last: bool = True
    end_of_stream: bool = False  # review raised because the video ended on an unconfirmed change


@dataclass
class TrackerConfig:
    stable_seconds: float = 1.0
    min_frames: int = 3
    dropout_piece_ratio: float = 0.5
    uncertain_threshold: float = 0.6
    margin_threshold: float = 3.0
    # A stable window may differ from its first frame on this many squares (detector flicker).
    window_tolerance: int = 1
    # The board counts as changed when at least this many confident squares disagree with the
    # confirmed position. 1 keeps moves whose other square is hidden by a hand; a lone misread
    # then only costs one stable window that the resolver closes as "no_move".
    deviation_squares: int = 1
    match: str = "auto"  # passed to the resolver: "auto" | "piece" | "colour"
    # Multi-move resolution for fast play (several moves between two stable views).
    max_plies: int = 3  # longest sequence searched when one move does not explain the board
    auto_multi_ply: bool = True  # commit a sequence automatically when it is unique and clear
    glimpse_min_frames: int = 2  # clear frames needed to keep a too-short view as ordering evidence
    max_misreads: int = 4  # most persistent misread squares learned from one confirmation


@dataclass
class TemporalTracker:
    board: chess.Board
    config: TrackerConfig = field(default_factory=TrackerConfig)
    state: TrackerState = TrackerState.STABLE
    window: list[FrameEvidence] = field(default_factory=list)
    ignored_signatures: set = field(default_factory=set)
    pending_review: TrackerEvent | None = None
    last_t: float | None = None
    transitions: list[tuple[float, str, str]] = field(default_factory=list)
    glimpses: list[BoardObservation] = field(default_factory=list)
    misreads: dict[int, str] = field(default_factory=dict)  # square -> persistent wrong reading

    def _learn_misreads(self, evidence: BoardObservation, changed: set[int]) -> None:
        """After a confirmation, remember squares the detector keeps reading wrongly.

        A misread is a confidently observed square that disagrees with the newly confirmed
        position. Entries for squares changed by the confirmed move(s) are dropped first. Nothing
        is learned if the disagreement is large (that points to a wrong position, not misreads).
        """
        colour = self._colour_level(evidence)
        kept = {sq: v for sq, v in self.misreads.items() if sq not in changed}
        fresh: dict[int, str] = {}
        for sq in chess.SQUARES:
            lbl = self._label(evidence.squares[sq], colour)
            if lbl in (None, "?"):
                continue
            p = self.board.piece_at(sq)
            exp = (EMPTY if p is None else ("W" if p.color else "B")) if colour else (p.symbol() if p else EMPTY)
            if lbl != exp:
                fresh[sq] = lbl
            else:
                kept.pop(sq, None)
        if len(fresh) <= self.config.max_misreads:
            kept.update(fresh)
        self.misreads = kept

    def _changed_by(self, before: chess.Board) -> set[int]:
        return {sq for sq in chess.SQUARES if before.piece_at(sq) != self.board.piece_at(sq)}

    def _mask_misreads(self, evidence: BoardObservation) -> BoardObservation:
        """Copy of ``evidence`` with squares showing a known misreading marked unknown.

        Stored with committed moves, so later revalidation (game edits) sees the same evidence
        the tracker used, without needing the tracker's misread memory.
        """
        if not self.misreads:
            return evidence
        colour = self._colour_level(evidence)
        squares = list(evidence.squares)
        masked = []
        for sq, want in self.misreads.items():
            s = squares[sq]
            if self._label(s, colour) == want:
                squares[sq] = s.model_copy(update={"probs": None, "top": None, "top_prob": 0.0, "occluded": True,
                                                   "uncertain": True, "reasons": ["known_misread"]})
                masked.append(chess.square_name(sq))
        if not masked:
            return evidence
        warnings = [*evidence.warnings, "ignored persistent misreads on " + ", ".join(sorted(masked))]
        return evidence.model_copy(update={"squares": squares, "warnings": warnings})

    # ---- helpers ------------------------------------------------------------
    def _set_state(self, t: float, new: TrackerState) -> None:
        if new != self.state:
            self.transitions.append((t, self.state.value, new.value))
            self.state = new

    def _colour_level(self, obs: BoardObservation) -> bool:
        m = self.config.match
        return m == "colour" or (m == "auto" and obs.source == "model")

    def _label(self, s, colour: bool) -> str | None:
        """Comparable label of a square: None if unknown, else piece label or colour class."""
        if s.occluded or s.top is None or s.probs is None:
            return None
        if colour:
            p: dict[str, float] = {EMPTY: 0.0, "W": 0.0, "B": 0.0}
            for lbl, v in s.probs.items():
                p[EMPTY if lbl == EMPTY else ("W" if lbl.isupper() else "B")] += v
            k = max(p, key=p.get)
            return k if p[k] >= self.config.uncertain_threshold else "?"
        return s.top if s.top_prob >= self.config.uncertain_threshold else "?"

    def _signature(self, obs: BoardObservation) -> tuple:
        colour = self._colour_level(obs)
        return tuple(self._label(s, colour) for s in obs.squares)

    @staticmethod
    def _distance(a: tuple, b: tuple) -> int:
        """Squares where both views are confident and disagree."""
        return sum(1 for x, y in zip(a, b) if x not in (None, "?") and y not in (None, "?") and x != y)

    def _confirmed_piece_count(self) -> int:
        return len(self.board.piece_map())

    def _is_dropout(self, ev: FrameEvidence) -> bool:
        if ev.observation is None:
            return True
        expected = self._confirmed_piece_count()
        return expected > 0 and ev.observation.piece_count() < self.config.dropout_piece_ratio * expected

    def _deviation_count(self, obs: BoardObservation) -> int:
        colour = self._colour_level(obs)
        n = 0
        for sq in chess.SQUARES:
            lbl = self._label(obs.squares[sq], colour)
            if lbl in (None, "?"):
                continue
            if self.misreads.get(sq) == lbl:
                continue  # known persistent misread, not a change
            p = self.board.piece_at(sq)
            if colour:
                exp = EMPTY if p is None else ("W" if p.color else "B")
            else:
                exp = p.symbol() if p else EMPTY
            if lbl != exp:
                n += 1
        return n

    def _deviates(self, obs: BoardObservation) -> bool:
        return self._deviation_count(obs) >= self.config.deviation_squares

    def _is_ignored(self, obs: BoardObservation) -> bool:
        sig = self._signature(obs)
        return any(self._distance(sig, ig) <= self.config.window_tolerance for ig in self.ignored_signatures)

    # ---- main entry ---------------------------------------------------------
    def step(self, ev: FrameEvidence) -> list[TrackerEvent]:
        if self.last_t is not None and ev.t <= self.last_t:
            return []  # ignore non-monotonic timestamps
        self.last_t = ev.t
        if self.state == TrackerState.NEEDS_REVIEW:
            return []
        blocked = ev.occluded or self._is_dropout(ev)

        if self.state == TrackerState.STABLE:
            if blocked:
                self._set_state(ev.t, TrackerState.OCCLUDED_OR_MOVING)
            elif self._deviates(ev.observation) and not self._is_ignored(ev.observation):
                self.window = [ev]
                self._set_state(ev.t, TrackerState.SETTLING)
            return []

        if self.state == TrackerState.OCCLUDED_OR_MOVING:
            if not blocked:
                self.window = [ev]
                self._set_state(ev.t, TrackerState.SETTLING)
                return self._maybe_validate()
            return []

        if self.state == TrackerState.SETTLING:
            if blocked:
                self._keep_glimpse()
                self.window = []
                self._set_state(ev.t, TrackerState.OCCLUDED_OR_MOVING)
                return []
            if self._distance(self._signature(ev.observation), self._consensus()) > self.config.window_tolerance:
                self._keep_glimpse()
                self.window = [ev]
            else:
                self.window.append(ev)
            return self._maybe_validate()
        return []

    def _consensus(self) -> tuple:
        """Per-square majority label of the current window (robust to single-frame flicker)."""
        sigs = [self._signature(f.observation) for f in self.window]
        out = []
        for sq in range(64):
            counts: dict = {}
            for s in sigs:
                counts[s[sq]] = counts.get(s[sq], 0) + 1
            out.append(max(counts, key=counts.get))
        return tuple(out)

    def _keep_glimpse(self) -> None:
        """Remember a clear but too-short view; it helps order moves made in quick succession."""
        if self.window and len(self.window) >= self.config.glimpse_min_frames:
            self.glimpses.append(aggregate([f.observation for f in self.window], self.config.uncertain_threshold))
            self.glimpses = self.glimpses[-12:]

    def _maybe_validate(self) -> list[TrackerEvent]:
        w = self.window
        if len(w) < self.config.min_frames or (w[-1].t - w[0].t) < self.config.stable_seconds:
            return []
        t0, t1 = w[0].t, w[-1].t
        self._set_state(t1, TrackerState.VALIDATING)
        evidence = aggregate([f.observation for f in w], self.config.uncertain_threshold)
        evidence = self._mask_misreads(evidence)
        result = resolve(self.board, evidence, uncertain_threshold=self.config.uncertain_threshold,
                         margin_threshold=self.config.margin_threshold, match=self.config.match,  # type: ignore[arg-type]
                         ignore=self.misreads)
        ref = w[0].ref
        self.window = []
        if result.status == "no_move":
            self.glimpses = []
            self._learn_misreads(evidence, set())
            self._set_state(t1, TrackerState.STABLE)
            return [TrackerEvent("no_move", t0, t1, result, evidence, ref)]
        if result.status == "confirmed" and result.best and result.best.uci:
            self.glimpses = []
            before = self.board.copy(stack=False)
            self.board.push_uci(result.best.uci)
            self._learn_misreads(evidence, self._changed_by(before))
            self._set_state(t1, TrackerState.STABLE)
            return [TrackerEvent("move", t0, t1, result, evidence, ref, uci=result.best.uci)]
        sequence = None
        if self.config.max_plies > 1:
            sequence = resolve_sequence(self.board, evidence, max_plies=self.config.max_plies,
                                        uncertain_threshold=self.config.uncertain_threshold,
                                        margin_threshold=self.config.margin_threshold,
                                        match=self.config.match,  # type: ignore[arg-type]
                                        glimpses=self.glimpses, ignore=self.misreads)
            if (self.config.auto_multi_ply and sequence.status == "confirmed" and sequence.best
                    and len(sequence.best.ucis) > 1):
                self.glimpses = []
                before = self.board.copy(stack=False)
                events = []
                for i, uci in enumerate(sequence.best.ucis):
                    last = i == len(sequence.best.ucis) - 1
                    self.board.push_uci(uci)
                    events.append(TrackerEvent("move", t0, t1, result, evidence, ref, uci=uci,
                                               sequence=sequence, sequence_index=i, sequence_last=last))
                self._learn_misreads(evidence, self._changed_by(before))
                self._set_state(t1, TrackerState.STABLE)
                return events
        event = TrackerEvent("review", t0, t1, result, evidence, ref, sequence=sequence)
        self.pending_review = event
        self._set_state(t1, TrackerState.NEEDS_REVIEW)
        return [event]

    def finish(self) -> list[TrackerEvent]:
        """End of the stream: if the last clear view still differs from the confirmed position, ask.

        The final view may be too short to count as stable, so it is never committed
        automatically. It is resolved for suggestions and returned as a ``review`` event.
        """
        if self.state == TrackerState.NEEDS_REVIEW:
            return []
        frames = self.window if self.window else []
        if frames:
            evidence = aggregate([f.observation for f in frames], self.config.uncertain_threshold)
            t0, t1, ref = frames[0].t, frames[-1].t, frames[0].ref
        elif self.glimpses:
            evidence, t0, t1, ref = self.glimpses[-1], self.last_t or 0.0, self.last_t or 0.0, None
        else:
            return []
        evidence = self._mask_misreads(evidence)
        if not self._deviates(evidence):
            return []
        result = resolve(self.board, evidence, uncertain_threshold=self.config.uncertain_threshold,
                         margin_threshold=self.config.margin_threshold, match=self.config.match,  # type: ignore[arg-type]
                         ignore=self.misreads)
        if result.status == "no_move":
            return []
        sequence = resolve_sequence(self.board, evidence, max_plies=self.config.max_plies,
                                    uncertain_threshold=self.config.uncertain_threshold,
                                    margin_threshold=self.config.margin_threshold,
                                    match=self.config.match,  # type: ignore[arg-type]
                                    glimpses=self.glimpses, ignore=self.misreads) if self.config.max_plies > 1 else None
        self.window = []
        event = TrackerEvent("review", t0, t1, result, evidence, ref, sequence=sequence, end_of_stream=True)
        self.pending_review = event
        self._set_state(t1, TrackerState.NEEDS_REVIEW)
        return [event]

    def resolve_review(self, uci: str | list[str] | None) -> None:
        """Apply the user's decision for the pending review.

        ``uci`` = a legal move (or a list of consecutive legal moves) from the confirmed position,
        or None for "no move happened" (the reviewed view is then ignored until it changes).
        """
        if self.state != TrackerState.NEEDS_REVIEW or self.pending_review is None:
            raise RuntimeError("no review pending")
        if uci is not None:
            ucis = [uci] if isinstance(uci, str) else list(uci)
            trial = self.board.copy()
            trial_start = self.board.copy(stack=False)
            for u in ucis:
                move = chess.Move.from_uci(u)
                if move not in trial.legal_moves:
                    raise ValueError(f"{u} is not legal in {trial.fen()}")
                trial.push(move)
            for u in ucis:
                self.board.push_uci(u)
            # Learn misreads only if the user's moves fully explain the reviewed view; otherwise the
            # view also shows later moves, which must not be mistaken for detector errors.
            ev = self.pending_review.evidence
            check = resolve(self.board, ev, uncertain_threshold=self.config.uncertain_threshold,
                            margin_threshold=self.config.margin_threshold, match=self.config.match,  # type: ignore[arg-type]
                            ignore=self.misreads)
            none_c = next((c for c in check.candidates if c.uci is None), None)
            base = len(none_c.mismatched_squares) if none_c else None
            # A real, not-yet-entered move shows up as a move that removes two or more mismatches
            # (it explains both its from- and to-square). Isolated disagreements are misreads.
            real_move_left = base is None or any(base - len(c.mismatched_squares) >= 2
                                                 for c in check.candidates if c.uci is not None)
            if not real_move_left:
                self._learn_misreads(ev, self._changed_by(trial_start))
            else:
                self.misreads = {sq: v for sq, v in self.misreads.items()
                                 if sq not in self._changed_by(trial_start)}
        else:
            self.ignored_signatures.add(self._signature(self.pending_review.evidence))
        self.glimpses = []
        self.pending_review = None
        self._set_state(self.last_t or 0.0, TrackerState.STABLE)
