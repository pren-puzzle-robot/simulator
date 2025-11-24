from __future__ import annotations
from typing import Iterable, List

from .point import Point
from .polygon import Polygon
from .piece_analysis import PieceType, OuterEdge, PieceAnalysis, analyze_polygon


class PuzzlePiece:
    """
    Represents a single puzzle piece.

    The piece is defined by:
    - a polygon (constructed from a list of Point instances)
    - a type (corner or edge)
    - a list of detected outer edges
    """

    _polygon: Polygon
    _type: PieceType
    _outer_edges: List[OuterEdge]

    def __init__(self, points: Iterable[Point]) -> None:
        points_list: List[Point] = list(points)
        if len(points_list) < 3:
            raise ValueError("PuzzlePiece requires at least 3 points")

        self._polygon = Polygon(points_list)

        # First analysis
        analysis = analyze_polygon(self._polygon)
        # Normalize vertex order based on this analysis
        self._normalize_vertex_order(analysis)

        # Re-analyze after rotation so indices and outer_edges match
        final_analysis = analyze_polygon(self._polygon)
        self._type = final_analysis.piece_type
        self._outer_edges = final_analysis.outer_edges

    def _normalize_vertex_order(self, analysis: PieceAnalysis) -> None:
        """
        Rotate polygon vertices so that the first vertex is the
        end point (j) of a chosen outer edge.
        """
        if not analysis.outer_edges:
            # Should not happen with only corner/edge pieces,
            # but do nothing if it does.
            return

        # Choose the last outer edge as canonical
        edge = analysis.outer_edges[-1]
        target_index = edge.j  # last vertex index of that edge

        verts = self._polygon.vertices
        n = len(verts)
        if n == 0:
            return

        # Rotate list: new_verts[0] == verts[target_index]
        target_index = target_index % n
        new_verts = verts[target_index:] + verts[:target_index]

        # Replace polygon with rotated vertices
        self._polygon = Polygon(new_verts)

    def get_triplet(self, index: int, direction: bool) -> tuple[Point, Point, Point]:
        """
        Get the point at the given `index` but also the two surounding\n
        points before and after. Works like a wrapper. After the last\n
        in the list comes the first again. The boolean says if you are\n
        going forward or backwards. Example `direction` == `true` -> forward,\n
        `false` -> backwards.
        """
        points: list[Point] = self.polygon.vertices
        length: int = len(points)
        inverted: int = 1 if direction else -1

        prev_index: int = (index - 1 * inverted) % length
        next_index: int = (index + 1 * inverted) % length

        return (points[prev_index], points[index], points[next_index])

    def get_limits(self) -> tuple[int, int]:
        """
        Get the lowest and the highest index of the polygon that are at\n
        the edge of an outer edge.
        """
        data: list[tuple[int, int]] = [edge.get_indices for edge in self._outer_edges]
        length: int = len(self.polygon.vertices)

        vertices: set[int] = {v for u, v in data for v in (u, v)}

        points_sorted = sorted(vertices)

        def cyclic_distance(a: int, b: int, n: int) -> int:
            """shortest distance in cyclic polygon"""
            d = (b - a) % n
            return d

        best_start = None
        best_len = None

        # try every starting index on the set
        for s in points_sorted:
            # distance to all the points
            dists = [cyclic_distance(s, v, length) for v in vertices]
            max_dist = max(dists)  # how far does it go
            if best_len is None or max_dist < best_len:
                best_len = max_dist
                best_start = s

        if best_start is None or best_len is None:
            raise RuntimeError("Internal error: no start point selected")

        start = best_start
        end = (best_start + best_len) % length

        result = (start, end)
        if start > end:
            result = (end, start)

        return result

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def type(self) -> PieceType:
        return self._type

    @property
    def outer_edges(self) -> List[OuterEdge]:
        """Detected outer edges of this piece."""
        return self._outer_edges

    @property
    def is_corner(self) -> bool:
        return self._type == PieceType.CORNER

    @property
    def is_edge(self) -> bool:
        return self._type == PieceType.EDGE

    def __repr__(self) -> str:
        return f"PuzzlePiece(type={self._type.value!r}, polygon={self._polygon!r})"
