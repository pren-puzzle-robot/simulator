from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from .point import Point
from .polygon import Polygon


# ---------- Tunables ----------
LONG_EDGE_VS_MEDIAN = 1.8  # edge is "long" if length >= median*this
LONG_EDGE_VS_MAX = 0.60  # ...and also >= this * (longest length)
CORNER_ANGLE_RANGE = (60.0, 120.0)  # degrees
# ------------------------------


class PieceType(str, Enum):
    CORNER = "corner"
    EDGE = "edge"  # default if not clearly a corner


@dataclass
class OuterEdge:
    """Outer edge of a piece, expressed in terms of polygon vertex indices."""

    i: int  # start vertex index
    j: int  # end vertex index (wrapped)
    p1: Point  # start point
    p2: Point  # end point
    length: float  # Euclidean length

    @property
    def get_indices(self) -> tuple[int, int]:
        """Return the indices of the outeredge."""
        return (self.i, self.j)
    
    def rotated(self, angle_rad: float, center: Point) -> OuterEdge:
        """Return a new OuterEdge rotated around center by angle_rad."""
        def rotate_point(p: Point, angle: float, center: Point) -> Point:
            s = math.sin(angle)
            c = math.cos(angle)

            # translate point back to origin:
            p_translated_x = p.x - center.x
            p_translated_y = p.y - center.y

            # rotate point
            x_new = p_translated_x * c - p_translated_y * s
            y_new = p_translated_x * s + p_translated_y * c

            # translate point back:
            x_final = x_new + center.x
            y_final = y_new + center.y

            return Point(x_final, y_final)

        new_p1 = rotate_point(self.p1, angle_rad, center)
        new_p2 = rotate_point(self.p2, angle_rad, center)
        new_length = math.hypot(new_p2.x - new_p1.x, new_p2.y - new_p1.y)

        return OuterEdge(i=self.i, j=self.j, p1=new_p1, p2=new_p2, length=new_length)


@dataclass
class PieceAnalysis:
    """Result of analyzing a polygon."""

    piece_type: PieceType
    outer_edges: List[OuterEdge]


def _to_xy(v) -> Tuple[float, float]:
    """Convert Point to (x, y) tuple of floats."""
    return float(v.x), float(v.y)


def _seg_len(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(math.hypot(dx, dy))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    num = float(np.dot(v1, v2))
    den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    a = math.degrees(math.acos(np.clip(num / den, -1.0, 1.0)))
    return a


def _edges_from_polygon(poly: Polygon) -> List[OuterEdge]:
    verts_raw = list(poly.vertices)
    n = len(verts_raw)
    edges: List[OuterEdge] = []

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = _to_xy(verts_raw[i])
        x2, y2 = _to_xy(verts_raw[j])
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        length = _seg_len((x1, y1), (x2, y2))
        edges.append(OuterEdge(i=i, j=j, p1=p1, p2=p2, length=length))

    return edges


def analyze_polygon(poly: Polygon) -> PieceAnalysis:
    """
    Decide if a polygon is a corner or an edge piece and return the outer edge(s).

    Rules:
    - If two long edges share a vertex and form a suitable angle -> CORNER (2 edges).
    - Else -> EDGE with one best outer edge.
      * If there are "long" edges, take the longest one.
      * If not, just take the globally longest edge.
    """
    edges = _edges_from_polygon(poly)
    if not edges:
        # In practice should not happen (Polygon requires >=3 vertices),
        # but we still return something valid.
        return PieceAnalysis(piece_type=PieceType.EDGE, outer_edges=[])

    lengths = np.array([e.length for e in edges], dtype=float)
    Lmax = float(lengths.max())
    Lmed = float(np.median(lengths))

    # candidate long edges
    long_edges: List[OuterEdge] = [
        e
        for e in edges
        if e.length >= max(LONG_EDGE_VS_MEDIAN * Lmed, LONG_EDGE_VS_MAX * Lmax)
    ]

    # Try to find a corner (two long edges sharing a vertex with angle in range)
    best_pair: tuple[OuterEdge, OuterEdge] | None = None
    best_sum = -1.0

    for a in long_edges:
        for b in long_edges:
            if a is b:
                continue

            shared_idx = None
            if a.i == b.j:
                shared_idx = a.i
                pa = np.array([a.p2.x - a.p1.x, a.p2.y - a.p1.y])
                pb = np.array([b.p1.x - b.p2.x, b.p1.y - b.p2.y])
            elif a.j == b.i:
                shared_idx = a.j
                pa = np.array([a.p1.x - a.p2.x, a.p1.y - a.p2.y])
                pb = np.array([b.p2.x - b.p1.x, b.p2.y - b.p1.y])
            elif a.i == b.i:
                shared_idx = a.i
                pa = np.array([a.p2.x - a.p1.x, a.p2.y - a.p1.y])
                pb = np.array([b.p2.x - b.p1.x, b.p2.y - b.p1.y])
            elif a.j == b.j:
                shared_idx = a.j
                pa = np.array([a.p1.x - a.p2.x, a.p1.y - a.p2.y])
                pb = np.array([b.p1.x - b.p2.x, b.p1.y - b.p2.y])

            if shared_idx is None:
                continue

            ang = _angle_between(pa, pb)
            if CORNER_ANGLE_RANGE[0] <= ang <= CORNER_ANGLE_RANGE[1]:
                s = a.length + b.length
                if s > best_sum:
                    best_sum = s
                    best_pair = (a, b)

    # Corner case found
    if best_pair is not None:
        e1, e2 = best_pair
        if e1.j == e2.i:
            ordered = [e1, e2]
        elif e2.j == e1.i:
            ordered = [e2, e1]
        else:
            ordered = [e1, e2]
        return PieceAnalysis(
            piece_type=PieceType.CORNER,
            outer_edges=ordered,
        )

    # No corner: it is an edge piece.
    # Prefer a long edge if available, otherwise just take the longest edge.
    if long_edges:
        best_edge = max(long_edges, key=lambda x: x.length)
    else:
        best_edge = max(edges, key=lambda x: x.length)

    return PieceAnalysis(
        piece_type=PieceType.EDGE,
        outer_edges=[best_edge],
    )
