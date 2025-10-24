# -*- coding: utf-8 -*-
"""
Base class for puzzle pieces and the values associated with them.\n
It is just an alterativ to the json files but it should make it\n
easier to access. Adapt at your own leisure.\n
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."


from .edge import Edge
from .corner import Corner

from pathlib import Path

import json


class PuzzlePiece:
    """A single puzzle piece that was recognized\n
    in the image and all the data associated with it."""

    JSON_EDGEDIR_NAME: str = "sides"
    JSON_SIDE_NAMES: list[str] = ["Top", "Right", "Bottom", "Left"]
    JSON_CORNER_NAME: str = "segment"
    JSON_ELEVATION_NAME: str = "signature"

    _top: Edge
    _right: Edge
    _bottom: Edge
    _left: Edge

    def __init__(self, top: Edge, right: Edge, bottom: Edge, left: Edge) -> None:
        self._top = top
        self._right = right
        self._bottom = bottom
        self._left = left

    @classmethod
    def from_json(cls, path: Path) -> PuzzlePiece:
        """Create a new PuzzlePiece from the information saved\n
        in the JSON file at the given directory via '''dir'''."""

        # open the json file at the given directory and store the data
        with open(path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        edge_data = json_data[cls.JSON_EDGEDIR_NAME]
        edges: dict[str, Edge] = {}

        for side in cls.JSON_SIDE_NAMES:
            signature_values: list[float] = edge_data[side][cls.JSON_ELEVATION_NAME]
            corner_values = edge_data[side][cls.JSON_CORNER_NAME]

            x1, y1 = corner_values[0]
            start = Corner(x=x1, y=y1)

            x2, y2 = corner_values[1]
            end = Corner(x=x2, y=y2)

            edges[side] = Edge(
                start=start, end=end, direction=side, signature=signature_values
            )

        return cls(
            top=edges[cls.JSON_SIDE_NAMES[0]],
            right=edges[cls.JSON_SIDE_NAMES[1]],
            bottom=edges[cls.JSON_SIDE_NAMES[2]],
            left=edges[cls.JSON_SIDE_NAMES[3]],
        )

    def __str__(self) -> str:
        return (
            "PuzzlePiece:\n"
            f" {str(self._top)}\n"
            f" {str(self._right)}\n"
            f" {str(self._bottom)}\n"
            f" {str(self._left)}\n"
        )
