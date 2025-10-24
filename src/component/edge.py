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

    def compute_similarity(self, other: Edge) -> float:
        """compute similarity between this edge and another one based on their signatures"""
        if len(self.get_signature) != len(other.get_signature):
            raise ValueError(
                "Signatures must be of the same length to compute similarity."
            )

        # simple similarity measure: inverse of mean absolute difference
        diffs = [
            abs(abs(a) - abs(b))
            for a, b in zip(self.get_signature, other.get_signature)
        ]
        mean_diff = sum(diffs) / len(diffs)

        # similarity score between 0 and 1
        similarity = max(0.0, 1.0 - mean_diff)
        return similarity
