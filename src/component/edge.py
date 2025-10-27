# -*- coding: utf-8 -*-
"""
Utility base class for Edges. It is supposed to make\n
the code more understandable, well organized and easier to\n
adapt in case something changes.
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."


from enum import Enum

from .corner import Corner


class EdgeDir(Enum):
    """state of information we have about an edge"""

    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    @staticmethod
    def from_str(label: str) -> EdgeDir:
        """convert a string to an EdgeDir"""
        label = label.upper()
        value: EdgeDir = EdgeDir[label]  # type: ignore

        if value.name in EdgeDir.__members__:
            return value

        raise ValueError(f"Unknown EdgeDir label: {label}")


class Edge:
    """Edge on a PuzzlePiece"""

    _start: Corner  # starting corner
    _end: Corner  # end corner
    _direction: EdgeDir  # direction to which the edge is facing
    _signature: list[float]  # signature of the edge

    def __init__(
        self, start: Corner, end: Corner, direction: str, signature: list[float]
    ) -> None:
        self._start = start
        self._end = end
        self._direction = EdgeDir.from_str(direction)
        self._signature = signature

    def __str__(self) -> str:
        return (
            f" {str(self._direction.name)} Edge:"
            f"({str(self._start)},{str(self._end)} \n {self._signature}"
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Edge):
            return False

        return (
            self.get_corners == value.get_corners
            and self.get_direction == value.get_direction
            and self.get_signature == value.get_signature
        )

    @property
    def get_corners(self) -> tuple[Corner, Corner]:
        """get both of the ``RealNode``'s that are connected to the edge"""
        return (self._start, self._end)

    @property
    def get_direction(self) -> str:
        """get a ``String`` representing the direction the edge is facing"""
        return self._direction.name

    @property
    def get_signature(self) -> list[float]:
        """get the signature of the edge"""
        return self._signature

    @property
    def get_width(self) -> float:
        """get the width of the edge"""
        return self._start.get_distance_between(self._end)

    @property
    def get_segment_length(self) -> float:
        """get the length of each segment in the signature"""
        return self.get_width / float(len(self._signature) - 1)

    def get_plotvalues(self) -> tuple[list[float], list[float]]:
        """get x and y values for plotting the signature"""
        x_values: list[float] = []
        y_values: list[float] = []

        width: float = self.get_segment_length

        length: int = len(self._signature)
        for i in range(length):
            x_values.append(i * width)
            y_values.append(self._signature[i])

        return (x_values, y_values)

    def get_integral(self) -> float:
        """Compute the integral of the edge's signature as a float value."""
        seg_length: float = self.get_segment_length

        integral: float = sum([abs(value) * seg_length for value in self.get_signature])
        return integral

    def get_local_middle_most_extrema(self) -> tuple[int, tuple[float, float]]:
        """Get the index and value of the middle left most local extrema (max or min) in the signature."""
        length: int = len(self._signature)
        mid_index: int = length // 2
        index: int = mid_index
        i: int = 0

        x, y = self.get_plotvalues()

        curr_extrema_index: int = index
        curr_extrema_value: float = y[index]

        index = index + i

        # search for local extrama by flickering outward from the middle
        while index >= 1 and index < length - 1:
            # check if next value is at least 5% bigger or further to the left
            if (
                abs(y[index]) >= abs(curr_extrema_value)
                and index < curr_extrema_index
                or abs(y[index]) >= abs(curr_extrema_value) * 1.05
            ):
                curr_extrema_index = index
                curr_extrema_value = y[index]

            # index flickering
            if i < 0:
                i = -i + 1
            else:
                i = -i

            # update index
            index = mid_index + i

    def compute_similarity(self, other: Edge) -> float:
        """Compute the similarity between this edge's signature and that of\n
        another by returning the percentile amount to which the integrals\n
        overlap and returning it as a `float` value between `0.0` and `1.0`."""
        if len(self.get_signature) != len(other.get_signature):
            raise ValueError(
                "Signatures must be of the same length to compute similarity."
            )

        mean_integral: float = (self_integral + other_integral) / 2.0

        # simple similarity measure: inverse of mean absolute difference
        diffs = [
            abs(a - b) * seg_length
            for a, b in zip(self.get_signature, other.get_signature)
        ]
        mean_diff = sum(diffs) / mean_integral
        similarity = max(0.0, 1.0 - mean_diff)
        return similarity
