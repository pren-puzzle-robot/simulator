from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np

from .point import Point
from .polygon import Polygon

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
        """Return the indices of the outer edge."""
        return self.i, self.j

    def rotated(self, angle_rad: float, center: Point) -> Edge:
        """Return a new Edge rotated around center by angle_rad."""
        def rotate_point(p: Point, angle: float) -> Point:
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

        new_p1 = rotate_point(self.p1, angle_rad)
        new_p2 = rotate_point(self.p2, angle_rad)
        new_length = math.hypot(new_p2.x - new_p1.x, new_p2.y - new_p1.y)

        return Edge(i=self.i, j=self.j, p1=new_p1, p2=new_p2, length=new_length)
    
    def translated(self, dx: float, dy: float) -> Edge:
        """Return a new Edge translated by (dx, dy)."""
        new_p1 = Point(self.p1.x + dx, self.p1.y + dy)
        new_p2 = Point(self.p2.x + dx, self.p2.y + dy)
        return Edge(i=self.i, j=self.j, p1=new_p1, p2=new_p2, length=self.length)

    def __repr__(self):
        return f"Edge(i={self.i}, j={self.j}, length={self.length:.2f}, \n\tp1={self.p1}, \n\tp2={self.p2})"

    def __hash__(self):
        return hash((self.i, self.j, self.p1.x, self.p1.y, self.p2.x, self.p2.y, self.length))
    

@dataclass
class OuterEdge:
    """Outer edge of a piece, expressed in terms of polygon vertex indices."""

    type: PieceType
    edges: List[Edge]

    @property
    def length(self) -> float:
        """Return the total length of the OuterEdge."""
        return sum(edge.length for edge in self.edges)

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

    # keep only edges that can face the outside (all points on same side as centroid and not multiple on same straight)
    outer_candidates = [
        e for e in edges
        if _edge_can_be_outer(e, poly)
    ]

    outer_candidates = _remove_lines_with_multiple_edges(outer_candidates)

    if not outer_candidates:
        return []

    chains = _build_edge_chains(outer_candidates)
    combos = _contiguous_edge_combos(chains)

    outer_edges: List[OuterEdge] = [OuterEdge(edges=c) for c in combos if c]
    outer_edges = sorted(outer_edges, key=lambda oe: -oe.length)

    return outer_edges



def _edge_can_be_outer(edge: Edge, poly: Polygon, relative_tolerance: float = 1e-2) -> bool:
    p1, p2 = edge.p1, edge.p2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = edge.length

    def signed_dist(q):
        # signed distance from q to the line p1->p2
        area2 = dx * (q.y - p1.y) - dy * (q.x - p1.x)
        return area2 / length

    # per-edge epsilon: grows with edge length
    eps = relative_tolerance * length

    s_centroid = signed_dist(poly.centroid())

    if abs(s_centroid) < eps:
        # centroid very close to the line => not a valid outer edge
        return False

    for v in poly.vertices:
        s_v = signed_dist(v)

        # points very close to the line are treated as "on" the line
        if abs(s_v) < eps:
            continue

        # if they have opposite signs and both are clearly off the line:
        if s_centroid * s_v < 0:
            return False

    return True

def _remove_lines_with_multiple_edges(edges: List[Edge]) -> List[Edge]:
    n = len(edges)
    to_remove = set()  # indices of edges to drop

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if _are_collinear(edges[i], edges[j]):
                # a second edge on same line => mark both and any others we find
                to_remove.add(i)
                to_remove.add(j)

    return [e for idx, e in enumerate(edges) if idx not in to_remove]


def _are_collinear(e1: Edge, e2: Edge,
                   angle_epsilon: float = 1e-3, # 0 => exactly parallel, 1 => not very parallel
                   distance_epsilon_percentage: float = 10) -> bool:
    p1, p2 = e1.p1, e1.p2
    q1, q2 = e2.p1, e2.p2

    a = np.array((p2.x, p2.y)) - np.array((p1.x, p1.y))  # dir of e1
    b = np.array((q2.x, q2.y)) - np.array((q1.x, q1.y))  # dir of e2

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        # degenerate segment(s)
        return False

    # 1) Check they are parallel (same or opposite direction)
    dot = np.dot(a, b) / (na * nb)   # cos(angle)
    if abs(abs(dot) - 1.0) > angle_epsilon:
        return False

    distance_epsilon = (e1.length + e2.length) / 2.0 / distance_epsilon_percentage

    # 2) Check they lie on the same line:
    # distance from q1 to the line through p1–p2
    v = np.array((q1.x, q1.y)) - np.array((p1.x, p1.y))
    cross = a[0] * v[1] - a[1] * v[0]          # 2D "z-component" of cross product
    distance = abs(cross) / na                 # perpendicular distance

    return distance < distance_epsilon




def _build_edge_chains(edges: List[Edge]) -> List[List[Edge]]:
    """Group edges into ordered chains by connectivity (e.j == next.i)."""
    if not edges:
        return []

    # Maps vertex index -> edge
    by_start = {e.i: e for e in edges}  # outgoing edges
    by_end   = {e.j: e for e in edges}  # incoming edges

    chains: List[List[Edge]] = []
    visited = set()

    for edge in edges:
        if edge in visited:
            continue

        # walk backwards to the start of this chain
        current = edge
        while current.i in by_end and by_end[current.i] is not current and by_end[current.i] not in visited:
            current = by_end[current.i]

        # walk forwards to build the full chain
        chain: List[Edge] = []
        while current not in visited:
            chain.append(current)
            visited.add(current)

            if current.j in by_start and by_start[current.j] not in visited:
                current = by_start[current.j]
            else:
                break

        chains.append(chain)

    return chains

def _contiguous_edge_combos(chains: List[List[Edge]]) -> List[List[Edge]]:
    """From chains of edges, build all contiguous sub-chains (combos)."""
    combos: List[List[Edge]] = []

    for chain in chains:
        n = len(chain)
        for start in range(n):
            for end in range(start, n):
                combos.append(chain[start:end+1])

    return combos