# -*- coding: utf-8 -*-
"""
Base class for puzzle pieces and the values associated with them.\n
It is just an alterativ to the json files but it should make it\n
easier to access. Adapt at your own leisure.\n
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN 1 Team 13, HS25. All rights reserved."

from pathlib import Path

import json

from .edge import Edge
from .corner import Corner
from utilities import Point


class PuzzlePiece:
    """A single puzzle piece that was recognized\n
    in the image and all the data associated with it."""

    JSON_INDEX_NAME: str = "piece"
    JSON_EDGEDIR_NAME: str = "sides"
    JSON_SIDE_NAMES: list[str] = ["Top", "Right", "Bottom", "Left"]
    JSON_CLASS_NAME: str = "class"
    JSON_CORNER_NAME: str = "segment"
    JSON_ELEVATION_NAME: str = "signature"

    _idx: int
    _top: Edge
    _right: Edge
    _bottom: Edge
    _left: Edge

    def __init__(
        self, idx: int, top: Edge, right: Edge, bottom: Edge, left: Edge
    ) -> None:
        self._idx = idx
        self._top = top
        self._right = right
        self._bottom = bottom
        self._left = left

    # mock
    def get_polygon(self) -> list[Point]:
        """mock method to trick pylance"""
        mock_list: list[Point] = []
        return mock_list

    # mock
    def get_triplet(self, index: int) -> tuple[Point, Point, Point]:
        """mock method returning the point at that index the previous one and the next"""
        a: Point = Point(0, 0)
        return (a, a, a)

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
    def get_edges(self) -> dict[str, Edge]:
        """Get all edges of the puzzle piece as a dictionary."""
        return {
            "Top": self._top,
            "Right": self._right,
            "Bottom": self._bottom,
            "Left": self._left,
        }

    def __str__(self) -> str:
        return (
            "PuzzlePiece:\n"
            f" Piece {str(self._idx)}\n"
            f" {str(self._top)}\n"
            f" {str(self._right)}\n"
            f" {str(self._bottom)}\n"
            f" {str(self._left)}\n"
        )
