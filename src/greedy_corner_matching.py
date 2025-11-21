"""Code-File providing the matching of puzzle edges."""

from __future__ import annotations

import math

from component import PuzzlePiece
from utilities import Point



ERROR_MARCHING_LENGTH: float = 0.05
ERROR_MARCHING_ANGLE: float = 10  # degrees

PUZZLE: dict[int, PuzzlePiece] = {}


def main():
    """greedy matching"""

    # do something to save all puzzlepieces

    # find_matching_puzzle_piece()


def _find_matching_puzzle_piece(origin: PuzzlePiece,
                                remaining: dict[int, PuzzlePiece]) -> tuple[int, dict[int, PuzzlePiece]]:
    prev_remaining = remaining
    curr_remaining = remaining

    o_length: int = len(origin.get_polygon())

    for i in range(1, o_length + 1):
        o_points = origin.get_triplet(i)
        for j, r in prev_remaining.items():
            # normal
            r_points_normal = r.get_triplet(i)
            if _next_segment_matches(o_points, r_points_normal):
                continue

            # reverse
            mirror_index: int = len(r.get_polygon())
            r_points_reverse = r.get_triplet(mirror_index)
            if _next_segment_matches(o_points, r_points_reverse):
                continue

            del curr_remaining[j]
        
        if len(curr_remaining.items()) == 0:
            # in this round all the pieces drop out
            return (i-1, prev_remaining)
        
        # there are still pieces in the pool
        prev_remaining = curr_remaining

    # if this is ever used, mistakes were made
    return (0, remaining)

def _next_segment_matches(origin: tuple[Point, Point, Point],
                          match: tuple[Point, Point, Point]) -> bool:
    o_prev, o_this, o_next = origin
    m_prev, m_this, m_next = match

    # does length match
    o_length: float = ((o_this.x - o_next.x) ** 2 + (o_this.y - o_next.y) ** 2) ** 0.5
    m_length: float = ((m_this.x - m_next.x) ** 2 + (m_this.y - m_next.y) ** 2) ** 0.5

    err_margin_len: float = o_length * ERROR_MARCHING_LENGTH
    len_matches: bool = abs(o_length - m_length) <= err_margin_len

    # does angle match
    o_angle = _angle_at_this(o_prev, o_this, o_next)
    m_angle = _angle_at_this(m_prev, m_this, m_next)

    angle_matches: bool = ((180.0 - ERROR_MARCHING_ANGLE / 2.0 <= o_angle + m_angle) and \
        (o_angle + m_angle <= 180.0 + ERROR_MARCHING_ANGLE / 2.0)) \
        or abs(o_angle - m_angle) <= ERROR_MARCHING_ANGLE / 2.0

    return len_matches and angle_matches

def _angle_at_this(prev: Point, this: Point, follow: Point) -> float:
    # vectors leaving the point "this"
    v1_x: float = prev.x - this.x
    v1_y: float = prev.y - this.y
    v2_x: float = follow.x - this.x
    v2_y: float = follow.y - this.y

    # compute their length
    len1_sq: float = v1_x * v1_x + v1_y * v1_y
    len2_sq: float = v2_x * v2_x + v2_y * v2_y

    # edgecase if identical points
    if len1_sq == 0 or len2_sq == 0:
        return float("nan")

    dot = v1_x * v2_x + v1_y * v2_y
    cross = v1_x * v2_y - v1_y * v2_x  # Skalar in 2D

    # Unsigned angle: 0..180°
    angle_rad = math.atan2(abs(cross), dot)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

if __name__ == "__main__":
    main()
