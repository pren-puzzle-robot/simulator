# -*- coding: utf-8 -*-
"""
Utility base class for Edges. It is supposed to make\n
the code more understandable, well organized and easier to\n
adapt in case something changes.
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."


class Corner:
    """A corner to the edge of puzzlepiece"""

    _x: float  # width / horizontal position / distance to upper left-corner of the image
    _y: float  # height / vertical position / distance to upper left-corner of the image

    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f"Corner: ({self._x} px,{self._y} px)"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Corner):
            return False

        return self.x == value.x and self.y == value.y

    @property
    def x(self) -> float:
        """get the x-coordinate of the corner"""
        return self.x

    @property
    def y(self) -> float:
        """get the y-coordinate of the corner"""
        return self.y

    def get_distance_between(self: Corner, other: Corner) -> float:
        """compute the distance between this corner and an other one"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
