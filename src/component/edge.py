# -*- coding: utf-8 -*-
"""
Utility base class for Edges. It is supposed to make\n
the code more understandable, well organized and easier to\n
adapt in case something changes.
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."


from enum import Enum

from numpy import interp, asarray

from .corner import Corner


class EdgeDir(Enum):
    """The direction said edge is facing on the picture."""

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


class EdgeCat(Enum):
    """A broad categorisation between flat edges at the\n
    exterior of the puzzle and edges that need to fit together."""

    HOLE = 0
    FLAT = 1

    @staticmethod
    def from_str(label: str) -> EdgeCat:
        """convert a string to an EdgeType"""
        label = label.upper()
        value: EdgeCat = EdgeCat[label]  # type: ignore

        if value.name in EdgeCat.__members__:
            return value

        raise ValueError(f"Unknown EdgeDir label: {label}")


class Edge:
    """Edge on a PuzzlePiece"""

    _piece: int  # index of the puzzle piece this edge belongs to
    _start: Corner  # starting corner
    _end: Corner  # end corner
    _direction: EdgeDir  # direction to which the edge is facing
    _cat: EdgeCat  # type of the edge
    _signature: list[float]  # signature of the edge

    def __init__(
        self,
        piece: int,
        start: Corner,
        end: Corner,
        direction: str,
        cat: str,
        signature: list[float],
    ) -> None:
        self._piece = piece
        self._start = start
        self._end = end
        self._direction = EdgeDir.from_str(direction)
        self._cat = EdgeCat.from_str(cat)
        self._signature = signature

    def __str__(self) -> str:
        return (
            f" {str(self._direction.name)} {str(self._cat)} Edge:"
            f"({str(self._start)},{str(self._end)} \n {self._signature}"
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Edge):
            return False

        return (
            self.get_piece == value.get_piece
            and self.get_corners == value.get_corners
            and self.get_direction == value.get_direction
            and self.get_cat == value.get_cat
            and self.get_signature == value.get_signature
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.get_piece,
                self.get_corners,
                self.get_direction,
                self.get_cat,
                tuple(self.get_signature),
            )
        )

    @property
    def get_piece(self) -> int:
        """get the index of the puzzle piece this edge belongs to"""
        return self._piece

    @property
    def get_corners(self) -> tuple[Corner, Corner]:
        """get both of the `Corner`'s that are connected to the edge"""
        return (self._start, self._end)

    @property
    def get_direction(self) -> str:
        """get a `String` representing the direction the edge is facing"""
        return self._direction.name

    @property
    def get_cat(self) -> str:
        """get a `String` representing the type of the edge"""
        return self._cat.name

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

    @property
    def get_height(self) -> float:
        """get the difference between the highest and lowest\n
        point in the signature of this edge"""
        return abs(max(self._signature) - min(self._signature))

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

        integral: float = sum(abs(value) * seg_length for value in self.get_signature)
        return integral

    def _get_off_set_between_signatures(self, other: Edge) -> tuple[float, float]:
        """Get the offset (index difference) between the middle left most\n
        local extrema (max or min) in the signatures of this edge and\n
        another edge."""
        self_value = self.get_local_middle_most_extrema()
        other_value = other.get_local_middle_most_extrema()

        value_offset: tuple[float, float] = (
            other_value[0] - self_value[0],
            other_value[1] - self_value[1],
        )

        return value_offset

    def get_local_middle_most_extrema(self) -> tuple[float, float]:
        """Get the index and value of the middle left most local extrema (max or min) in the signature."""
        length: int = len(self._signature)
        mid_index: int = length // 2
        index: int = mid_index
        i: int = 0

        x, y = self.get_plotvalues()

        curr_extrema_index: int = index
        curr_extrema_value: float = y[index]

        left_bound: float = x[0]
        right_bound: float = x[-1]

        index = index + i

        # search for local extrama by flickering outward from the middle
        while 1 <= index < length - 1:
            # check if next value is at least 5% bigger or further to the left
            if (
                abs(y[index]) * 1.04 >= abs(curr_extrema_value)
                and index < curr_extrema_index
                or abs(y[index]) >= abs(curr_extrema_value) * 1.05
            ):
                curr_extrema_index = index
                curr_extrema_value = y[index]
                # print(index)

            # index flickering
            if i <= 0:
                i = -i + 1
            else:
                i = -i

            # update index
            index = mid_index + i

            # exit loop if the found extrema is larger than the distance to the edge
            if (left_bound + abs(curr_extrema_value)) ** 2 >= x[index] or (
                right_bound - abs(curr_extrema_value)
            ) ** 2 <= x[index]:
                break

        return_value: tuple[float, float] = (x[curr_extrema_index], curr_extrema_value)

        return return_value

    def compute_similarity(self, other: Edge) -> float:
        """Compute the similarity between this edge's signature and that of\n
        another by returning the percentile amount to which the integrals\n
        overlap and returning it as a `float` value between `0.0` and `1.0`."""
        if len(self.get_signature) != len(other.get_signature):
            raise ValueError(
                "Signatures must be of the same length to compute similarity."
            )

        value_offset = self._get_off_set_between_signatures(other)

        mean_integral: float = (self.get_integral() + other.get_integral()) / 2.0
        mean_width: float = (self.get_width + other.get_width) / 2.0
        mean_height: float = (self.get_height + other.get_height) / 2.0

        diff_integral: float = self.compute_difference(other)

        fac_integral: float = 1.0 - diff_integral / mean_integral
        fac_width: float = 1.0 - abs(value_offset[0]) / mean_width
        fac_height: float = 1.0 - abs(value_offset[1]) / mean_height

        similarity = fac_integral * fac_width * fac_height

        return similarity

    def compute_difference(self, other: Edge) -> float:
        """Compute the difference between this edge's signature and that of\n
        another by returning the percentile amount to which the integrals\n
        differ and returning it as a `float` value between `0.0` and `1.0`."""

        x, y_values = self._compute_matching_plots(other)
        y_s, y_o = y_values

        seg_length: float = abs(x[0] - x[-1]) / float(len(x))

        sum_diffs_var1: float = 0.0
        sum_diffs_var2: float = 0.0

        for y1, y2 in zip(y_s, y_o):
            sum_diffs_var1 += abs(min(0.0, (y1 - y2))) * seg_length
            sum_diffs_var2 += abs(min(0.0, (y2 - y1))) * seg_length

        return min(sum_diffs_var1, sum_diffs_var2)

    def _compute_matching_plots(
        self, other: Edge
    ) -> tuple[list[float], tuple[list[float], list[float]]]:
        """Compute the matching plot values between this edge's signature and that of\n
        another edge, adjusted for offset."""
        value_offset = self._get_off_set_between_signatures(other)

        x_s, y_s = self.get_plotvalues()
        x_o, y_o = other.get_plotvalues()

        adjusted_x_s = [a + value_offset[0] for a in x_s]
        adjusted_y_s = [b + value_offset[1] for b in y_s]

        min_x_value: float = max(adjusted_x_s[0], x_o[0])
        max_x_value: float = min(adjusted_x_s[-1], x_o[-1])

        start_idx = next(i for i, v in enumerate(adjusted_x_s) if v >= min_x_value)
        end_idx = next(i for i, v in enumerate(adjusted_x_s) if v >= max_x_value)

        intersect_x = adjusted_x_s[start_idx:end_idx]

        intersect_y_s = adjusted_y_s[start_idx:end_idx]

        intersect_y_o = asarray(interp(intersect_x, x_o, y_o), dtype=float).tolist()

        return (intersect_x, (intersect_y_s, intersect_y_o))
