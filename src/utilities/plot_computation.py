# -*- coding: utf-8 -*-
"""
Utility class for all the computation of the plots generated\n
by the edges of the puzzle pieces. It is supposed to make\n
the code and the calculations behind it more understandable,\n
organized and easier to adapt or maintaine in case something\n
changes.
"""

from __future__ import annotations

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."

from itertools import permutations, combinations

import numpy as np

from scipy.signal import find_peaks

from .point import Point

MIN_PROMINENCE = 0.1
MIN_DISTANCE = 10


def compute_offset(
    plot_a: tuple[list[float], list[float]], plot_b: tuple[list[float], list[float]]
) -> tuple[float, float]:
    """
    Analyze a single (x, y) plot and return a list of peak points.
    """
    peaks_a = _get_peaks_in_plot(plot_a)
    peaks_b = _get_peaks_in_plot(plot_b)

    points_a = list(Point(x, y) for (x, y) in peaks_a)
    points_b = list(Point(x, y) for (x, y) in peaks_b)

    matching = _find_best_matching(points_a, points_b)

    offset = _calculate_offset_from_matching(matching)

    return offset


def _get_peaks_in_plot(
    plot: tuple[list[float], list[float]],
) -> list[tuple[float, float]]:
    """
    Analyze a single (x, y) plot and return a list of peak points.
    """
    x, y = plot
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float) * -1.0

    # Find peaks
    peaks, props = find_peaks(y_np, prominence=MIN_PROMINENCE, distance=MIN_DISTANCE)

    # Collect peak information
    points = list((float(x_np[i]), float(y_np[i] * -1.0)) for j, i in enumerate(peaks))

    return points


# Pubicliy accessable function that clients can use to find a matching. Gives access
# to all the other functions and simplifys access by hiding all the conditional
# statemates and adaptations. On its own only checks whether the two given lists as
# parameters have the same length and decides which recursive method needs to be
# invoked. Returns the best matching regardless with the nodes that could not be
# asigned as an empty tuple.
# - image   = (list[VisualNodes]) list of nodes extracted from the image
# - calc    = (list[VisualNodes]) list of nodes that were calculated
# - returns = (list[(str,str)]) list of tuples that create the minimal Matching
def _find_best_matching(
    peaks_a: list[Point], peaks_b: list[Point]
) -> list[tuple[Point, Point]]:
    """give it two list of VisualNodes, lean back and enjoy so magic happening"""
    result: list[tuple[Point, Point]] = []

    if len(peaks_a) != len(peaks_b):
        result = _create_subset_for_matching(peaks_a, peaks_b)
    else:
        _, result = _calculate_best_matching(peaks_a, peaks_b)

    return result


# Private function that does all the heavy lifting. Generates all possible
# permutations between the two sets and returns the minimal matching. Does
# however need the lists to be of equal length. Because of that it might be
# necessary use the method repeatedly. For that case it can return in addition
# to the matching the corresponding distance so the caller can compare results
# of different sublists.
# TODO: # pylint: disable=fixme
# Exceptionally slow since this part alone takes O(N!).
# Alternatives desperately needed.
# - image    = (list[VisualNodes]) list of nodes extracted from the image
# - calc     = (list[VisualNodes]) list of nodes that were calculated
# - needEval = (bool) caller can say if they want the distance as well
# - returns  = (list[(str,str)]) list of tuples that create the minimal Matching
# alternative:
# - returns  = (float, list[str,str]) same as above but with the distance
def _calculate_best_matching(peaks_a: list[Point], peaks_b: list[Point]):
    if len(peaks_a) != len(peaks_b):
        raise ValueError("Calculation Failed. The arrays don't have the same length.")

    perms = permutations(peaks_a)
    current_best = float("inf")
    result: list[tuple[Point, Point]] = []

    for p in perms:
        temp = _calculate_distance(list(p), peaks_b)
        if temp < current_best:
            current_best = temp
            result = _calculate_matching(list(p), peaks_b)

    return (current_best, result)


# Private function that is needed in cases the list do not have the same length.
# Generates subsets on all the possible subsets and recursively checks them all.
# TODO: # pylint: disable=fixme
# exceptionally slow since it makes the algorithm O(N!*N!) Alternatives
# desperately needed.
# - image   = (list[VisualNodes]) list of nodes extracted from the image
# - calc    = (list[VisualNodes]) list of nodes that were calculated
# - returns = (list[str,str]) list of tuples that create the minimal Matching
def _create_subset_for_matching(
    plot_a: list[Point], plot_b: list[Point]
) -> list[tuple[Point, Point]]:
    len_a: int = len(plot_a)
    len_b: int = len(plot_b)

    fix: list[Point] = []

    if len_a > len_b:
        comb = combinations((a for a in plot_a), len_b)
        fix = plot_b
    elif len_a < len_b:
        comb = combinations((b for b in plot_b), len_a)
        fix = plot_a
    else:
        raise ValueError(
            "Calculation went wrong. The arrays were already the same length"
            + " and did not need to be turned into subsets."
        )

    current_best = float("inf")
    result: list[tuple[Point, Point]] = []

    # try every subset
    for c in comb:
        temp_dis, temp_res = _calculate_best_matching(list(c), fix)
        if temp_dis < current_best:
            current_best = temp_dis
            result = temp_res

    return result


# Calculates the distance between each node in both arrays
# - image   = Array of VisualNodes
# - calc    = Array of VisualNodes
# - returns = (float) total distance
def _calculate_distance(peaks_a: list[Point], peaks_b: list[Point]) -> float:
    if len(peaks_a) != len(peaks_b):
        raise ValueError("Calculation Failed. The arrays don't have the same length.")

    value: float = 0.0
    match: list[tuple[Point, Point]] = list(zip(peaks_a, peaks_b))
    for a, b in match:
        value += a.get_distance_between(b)

    return value


# Generates a list (matching) of tuples for two arrays of nodes
# - image   = Array of Nodes
# - calc    = Array of Nodes
# - returns = Array of Tuples
def _calculate_matching(
    peaks_a: list[Point], peaks_b: list[Point]
) -> list[tuple[Point, Point]]:
    if len(peaks_a) != len(peaks_b):
        raise ValueError("Calculation Failed. The arrays don't have the same length.")

    match: list[tuple[Point, Point]] = list(zip(peaks_a, peaks_b))

    return match


def _calculate_offset_from_matching(
    matching: list[tuple[Point, Point]],
) -> tuple[float, float]:
    """Calculate the average offset from a given matching."""
    if not matching:
        return (0.0, 0.0)

    total_dx = 0.0
    total_dy = 0.0
    for a, b in matching:
        total_dx += a.x - b.x
        total_dy += a.y - b.y

    avg_dx = total_dx / len(matching)
    avg_dy = total_dy / len(matching)

    return (avg_dx, avg_dy)


def main() -> None:
    """Main function for quick testing purposes."""
    peaks_a = [Point(1.0, 2.0), Point(3.0, 4.0), Point(5.0, 6.0), Point(7.0, 8.0)]
    peaks_b = [Point(1.5, 2.5), Point(3.5, 4.5), Point(5.5, 6.5)]
    matching = _find_best_matching(peaks_a, peaks_b)
    print("Best Matching:", list(f"{a} -> {b}" for a, b in matching))


if __name__ == "__main__":
    main()
