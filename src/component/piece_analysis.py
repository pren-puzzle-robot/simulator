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
class Edge:
    """Edge of a piece, expressed in terms of polygon vertex indices."""

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
        """Return a new Edge rotated around center by angle_rad."""
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

        return Edge(i=self.i, j=self.j, p1=new_p1, p2=new_p2, length=new_length)
    
    def translated(self, dx: float, dy: float) -> Edge:
        """Return a new Edge translated by (dx, dy)."""
        new_p1 = Point(self.p1.x + dx, self.p1.y + dy)
        new_p2 = Point(self.p2.x + dx, self.p2.y + dy)
        return Edge(i=self.i, j=self.j, p1=new_p1, p2=new_p2, length=self.length)

    def __repr__(self):
        return f"Edge(i={self.i}, j={self.j}, length={self.length:.2f}, \n\tp1={self.p1}, \n\tp2={self.p2})"
    

@dataclass
class OuterEdge:
    """Outer edge of a piece, expressed in terms of polygon vertex indices."""

    type: PieceType
    edges: List[Edge]

    def rotated(self, angle_rad: float, center: Point) -> OuterEdge:
        """Return a new OuterEdge rotated around center by angle_rad."""
        rotated_edges = [edge.rotated(angle_rad, center) for edge in self.edges]
        return OuterEdge(edges=rotated_edges)
    
    def translated(self, dx: float, dy: float) -> OuterEdge:
        """Return a new OuterEdge translated by (dx, dy)."""
        translated_edges = [edge.translated(dx, dy) for edge in self.edges]
        return OuterEdge(edges=translated_edges)

    def __init__(self, edges: List[Edge]) -> None:
        self.edges = edges
        self.type = PieceType.EDGE if len(edges) == 1 else PieceType.CORNER

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


def _edges_from_polygon(poly: Polygon) -> List[Edge]:
    verts_raw = list(poly.vertices)
    n = len(verts_raw)
    edges: List[Edge] = []

    for i in range(n):
        j = (i + 1) % n
        x1, y1 = _to_xy(verts_raw[i])
        x2, y2 = _to_xy(verts_raw[j])
        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        length = _seg_len((x1, y1), (x2, y2))
        edges.append(Edge(i=i, j=j, p1=p1, p2=p2, length=length))

    return edges


def analyze_polygon(poly: Polygon) -> List[OuterEdge]:
    """
    Find all candidate outer-edge options for a piece.

    - Corner piece -> each option is an OuterEdge with two Edges (corner).
    - Edge piece   -> each option is an OuterEdge with one Edge.
    """
    edges = _edges_from_polygon(poly)
    if not edges:
        return []

    lengths = np.array([e.length for e in edges], dtype=float)
    Lmax = float(lengths.max())
    Lmed = float(np.median(lengths))

    # long edge candidates (same idea as before)
    long_edges: List[Edge] = [
        e
        for e in edges
        if e.length >= max(LONG_EDGE_VS_MEDIAN * Lmed, LONG_EDGE_VS_MAX * Lmax)
    ]

    corner_candidates: List[OuterEdge] = []

    for idx_a, a in enumerate(long_edges):
        for idx_b, b in enumerate(long_edges):
            if idx_b <= idx_a:
                continue  # avoid (a,b) and (b,a) duplicates and self-pair

            shared_idx = None

            # we also define pa, pb as vectors pointing away from the shared vertex
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
                # order edges so that they follow each other around the polygon
                if a.j == b.i:
                    ordered = [a, b]
                elif b.j == a.i:
                    ordered = [b, a]
                else:
                    ordered = [a, b]  # ambiguous but still a valid pair

                corner_candidates.append(OuterEdge(edges=ordered))

    # If we found any corner candidates, treat this as a corner piece
    if corner_candidates:
        # sort by total length descending
        corner_candidates.sort(
            key=lambda oe: sum(e.length for e in oe.edges),
            reverse=True,
        )
        return corner_candidates

    # ---------- Edge piece: give all single-edge options ----------
    if long_edges:
        candidate_edges = sorted(long_edges, key=lambda e: e.length, reverse=True)
    else:
        candidate_edges = sorted(edges, key=lambda e: e.length, reverse=True)

    return [OuterEdge(edges=[e]) for e in candidate_edges]
