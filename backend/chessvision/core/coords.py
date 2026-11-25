"""Canonical board coordinate conventions. Defined here once and used everywhere.

* Squares use python-chess indices: ``square = file + 8 * rank`` with file 0..7 = a..h
  and rank 0..7 = 1..8.
* The canonical warped board is ``size`` x ``size`` pixels with a8 at the top-left (0, 0),
  h8 at (size, 0), h1 at (size, size) and a1 at (0, size). White is at the bottom.
* Image coordinates are (x, y) with y pointing down.
"""

from __future__ import annotations

import math

import chess

FILES = "abcdefgh"


def square_name(square: int) -> str:
    return chess.square_name(square)


def parse_square(name: str) -> int:
    return chess.parse_square(name)


def square_at(x: float, y: float, size: float) -> int | None:
    """Square under canonical warped pixel (x, y), or None if outside the board. Never clamps."""
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if x < 0 or y < 0 or x >= size or y >= size:
        return None
    cell = size / 8.0
    file = int(x // cell)
    row = int(y // cell)
    return chess.square(file, 7 - row)


def square_bounds(square: int, size: float) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) of a square in the canonical warped board."""
    cell = size / 8.0
    file = chess.square_file(square)
    row = 7 - chess.square_rank(square)
    return file * cell, row * cell, (file + 1) * cell, (row + 1) * cell


def square_center(square: int, size: float) -> tuple[float, float]:
    x1, y1, x2, y2 = square_bounds(square, size)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def edge_distance_fraction(x: float, y: float, square: int, size: float) -> float:
    """Distance from (x, y) to the nearest border of ``square`` as a fraction of the square side."""
    x1, y1, x2, y2 = square_bounds(square, size)
    cell = size / 8.0
    return min(x - x1, x2 - x, y - y1, y2 - y) / cell
