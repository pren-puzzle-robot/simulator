from __future__ import annotations
from typing import Iterable, List

from .point import Point
from .polygon import Polygon
from .edge import Edge
from .outer_edge import OuterEdge, PieceType



class PuzzlePiece:
    """
    Represents a single puzzle piece.

    The piece is defined by:
    - a polygon (constructed from a list of Point instances)
    - a type (corner or edge)
    - a list of detected outer edges
    """

    _polygon: Polygon
    _possible_possible_outer_edges: List[OuterEdge]
    _outer_edge: OuterEdge

    _rotation: float = 0.0  # in radians
    _translation: tuple[float, float] = (0.0, 0.0)

    def __init__(self, points: Iterable[Point]) -> None:
        points_list: List[Point] = list(points)
        if len(points_list) < 3:
            raise ValueError("PuzzlePiece requires at least 3 points")

        self._translation: tuple[float, float] = (0.0, 0.0)
        self._polygon = Polygon(points_list)

        # First analysis
        # analysis = analyze_polygon(self._polygon)
        # # Normalize vertex order based on this analysis
        # self._normalize_vertex_order(analysis)

        # Re-analyze after rotation so indices and outer_edges match
        from utilities import analyze_polygon
        final_analysis = analyze_polygon(self._polygon)
        self._possible_outer_edges = final_analysis
        self._outer_edge = final_analysis[0]

    # def _normalize_vertex_order(self, analysis: PieceAnalysis) -> None:
    #     """
    #     Rotate polygon vertices so that the first vertex is the
    #     end point (j) of a chosen outer edge.
    #     """
    #     if not analysis.outer_edges:
    #         # Should not happen with only corner/edge pieces,
    #         # but do nothing if it does.
    #         return

    #     # Choose the last outer edge as canonical
    #     edge = analysis.outer_edges[-1]
    #     target_index = edge.j  # last vertex index of that edge

    #     verts = self._polygon.vertices
    #     n = len(verts)
    #     if n == 0:
    #         return

    #     # Rotate list: new_verts[0] == verts[target_index]
    #     target_index = target_index % n
    #     new_verts = verts[target_index:] + verts[:target_index]

    #     # Replace polygon with rotated vertices
    #     self._polygon = Polygon(new_verts)

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
        data: list[tuple[int, int]] = [edge.get_indices for edge in self.outer_edge.edges]
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
    
    def rotate(self, angle_rad: float) -> None:
        """Rotate the puzzle piece polygon by the given angle in radians."""
        self._rotation += angle_rad

        self._polygon.rotate(angle_rad)
        for edge in self._possible_outer_edges:
            edge.rotate(angle_rad, self.polygon.centroid())

    def translate(self, from_point: Point, to_point: Point) -> None:
        """Translate the puzzle piece so that from_point moves to to_point."""
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y

        self._translation= (self._translation[0] + dx, self._translation[1] + dy)
        print("Translation:", self._translation)

        self._polygon.translate(dx, dy)
        self._possible_outer_edges = [ edge.translated(dx, dy) for edge in self._possible_outer_edges ]
        self._outer_edge = self.outer_edge.translated(dx, dy)

    @property
    def polygon(self) -> Polygon:
        return self._polygon

    @property
    def type(self) -> PieceType:
        return self.outer_edge.type

    @property
    def possible_outer_edges(self) -> List[OuterEdge]:
        """Detected outer edges of this piece."""
        return self._possible_outer_edges
    
    @property
    def outer_edge(self) -> OuterEdge:
        """Outer edge of this piece."""
        return self._outer_edge

    @property
    def is_corner(self) -> bool:
        return self.outer_edge.type == PieceType.CORNER

    @property
    def is_edge(self) -> bool:
        return self.outer_edge.type == PieceType.EDGE
    
    @property
    def rotation(self) -> float:
        """Get the current rotation of the puzzle piece in radians."""
        return self._rotation
    
    @property
    def translation(self) -> tuple[float, float]:
        """Get the current translation (dx, dy) of the puzzle piece."""
        return self._translation

    def __repr__(self) -> str:
        return f"PuzzlePiece(polygon={self._polygon!r}, outer_edge={self.outer_edge})"
