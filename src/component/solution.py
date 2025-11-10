# -*- coding: utf-8 -*-
"""
Base class for puzzle pieces and the values associated with them.\n
It is just an alterativ to the json files but it should make it\n
easier to access. Adapt at your own leisure.\n
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN 1 Team 13, HS25. All rights reserved."

from enum import Enum
from pathlib import Path

import json


class _Side(str, Enum):
    """One of the four sides of a puzzle piece."""

    TOP = "Top"
    RIGHT = "Right"
    BOTTOM = "Bottom"
    LEFT = "Left"
    NONE = "None"

    @staticmethod
    def from_str(label: str) -> _Side:
        """convert a string to an `_Side`"""
        label = label.upper()
        value: _Side = _Side[label]  # type: ignore

        if value.name in _Side.__members__:
            return value

        raise ValueError(f"Unknown EdgeDir label: {label}")


class Solution:
    """A single puzzle piece that was recognized\n
    in the image and all the data associated with it."""

    JSON_NUMBER_OF_PIECES_KEY: str = "number_of_pieces"
    JSON_SOLUTION_KEY: str = "matches"
    JSON_SIDE_KEY: str = "sides"
    JSON_SIDE_NAMES: list[str] = ["Top", "Right", "Bottom", "Left"]

    JSON_MATCHED_PIECE_KEY: str = "matched_piece"
    JSON_MATCHED_PIECE_SIDE_KEY: str = "matched_side"

    _solution_registry: dict[int, dict[_Side, tuple[int, _Side]]]

    def __init__(self, solution: dict[int, dict[_Side, tuple[int, _Side]]]) -> None:
        self._solution_registry = solution

    @classmethod
    def from_json(cls, path: Path) -> Solution:
        """Create a new `Solution` object with a registry to\n
        save and easily access the informationa about a known\n
        solution for a given puzzle. The required parameter\n
        is the `path` to the `JSON` file."""

        # open the json file at the given directory and store the data
        with open(path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        n: int = json_data[cls.JSON_NUMBER_OF_PIECES_KEY]
        solution_data = json_data[cls.JSON_SOLUTION_KEY]

        result: dict[int, dict[_Side, tuple[int, _Side]]] = {}

        for i in range(1, n + 1):
            temp_piece_data = solution_data[str(i)][cls.JSON_SIDE_KEY]
            temp_piece_result: dict[_Side, tuple[int, _Side]] = {}
            for side in cls.JSON_SIDE_NAMES:
                temp_side_data = temp_piece_data[side]
                if (
                    temp_side_data[cls.JSON_MATCHED_PIECE_KEY] is not None
                    and temp_side_data[cls.JSON_MATCHED_PIECE_SIDE_KEY] is not None
                ):
                    temp_match_piece: int = temp_side_data[cls.JSON_MATCHED_PIECE_KEY]
                    temp_match_side: _Side = _Side(
                        temp_side_data[cls.JSON_MATCHED_PIECE_SIDE_KEY]
                    )
                    temp_piece_result[_Side(side)] = (temp_match_piece, temp_match_side)
                else:
                    temp_piece_result[_Side(side)] = (0, _Side("None"))
            result[i] = temp_piece_result

        return cls(result)

    def __str__(self) -> str:
        output: str = "Solution:\n"
        for piece, sides in self._solution_registry.items():
            output += f" Piece {piece}:\n"
            for side, match in sides.items():
                if self._has_match(piece, side.value):
                    output += f"  Side {side.value} -> ({match[0]}, {match[1].value})\n"
        return output

    def get_match(self, piece: int, side: str) -> tuple[int, str] | None:
        """Get the matching piece and side for the given\n
        `piece` and `side`. If no match exists, `None` is returned."""

        if self._has_match(piece, side):
            align = _Side.from_str(side)
            internal_result: tuple[int, _Side] = self._solution_registry[piece][align]
            return (internal_result[0], internal_result[1].value)

        return None

    def _has_match(self, piece: int, side: str) -> bool:
        """Check if there is a matching piece and side for the given\n
        `piece` and `side`."""

        try:
            align = _Side.from_str(side)
        except ValueError:
            return False

        internal_result: tuple[int, _Side] = self._solution_registry[piece][align]

        return internal_result[0] != 0 and internal_result[1] != _Side.NONE
