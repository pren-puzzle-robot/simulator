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
