from __future__ import annotations
from typing import Iterable, List

from pathlib import Path

import json

from .edge import Edge
from .corner import Corner


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

    def __init__(
        self, idx: int, top: Edge, right: Edge, bottom: Edge, left: Edge
    ) -> None:
        self._idx = idx
        self._top = top
        self._right = right
        self._bottom = bottom
        self._left = left

    @classmethod
    def from_json(cls, path: Path) -> PuzzlePiece:
        """Create a new `PuzzlePiece` from the information saved\n
        in the `JSON` file at the given directory via `dir`."""

        # open the json file at the given directory and store the data
        with open(path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        idx: int = int(json_data[cls.JSON_INDEX_NAME][-1])

        edge_data = json_data[cls.JSON_EDGEDIR_NAME]
        edges: dict[str, Edge] = {}

        for side in cls.JSON_SIDE_NAMES:
            signature_values: list[float] = edge_data[side][cls.JSON_ELEVATION_NAME]
            corner_values = edge_data[side][cls.JSON_CORNER_NAME]
            edge_class: str = edge_data[side][cls.JSON_CLASS_NAME]

            x1, y1 = corner_values[0]
            start = Corner(x=x1, y=y1)

            x2, y2 = corner_values[1]
            end = Corner(x=x2, y=y2)

            edges[side] = Edge(
                piece=idx,
                start=start,
                end=end,
                direction=side,
                cat=edge_class,
                signature=signature_values,
            )

        return cls(
            idx=idx,
            top=edges[cls.JSON_SIDE_NAMES[0]],
            right=edges[cls.JSON_SIDE_NAMES[1]],
            bottom=edges[cls.JSON_SIDE_NAMES[2]],
            left=edges[cls.JSON_SIDE_NAMES[3]],
        )

    @property
    def is_edge(self) -> bool:
        return self._type == PieceType.EDGE

    def __repr__(self) -> str:
        return f"PuzzlePiece(type={self._type.value!r}, polygon={self._polygon!r})"
