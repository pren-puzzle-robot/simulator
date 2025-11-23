"""Code-File providing the matching of puzzle edges."""

from __future__ import annotations

import math
import cv2

from component import PuzzlePiece, Point
from component.draw_puzzle_piece import render_puzzle_piece
from utilities import load_pieces

ERROR_MARCHING_LENGTH: float = 0.05
ERROR_MARCHING_ANGLE: float = 10  # degrees

PUZZLE: dict[int, PuzzlePiece] = load_pieces()


def main() -> None:
    """greedy matching"""

    # get the first corner piece
    starting_index, starting_piece = next(
        (i, p) for i, p in PUZZLE.items() if p.is_corner
    )
    remaining_pieces: dict[int, PuzzlePiece] = {
        k: v for k, v in PUZZLE.items() if k != starting_index
    }

    # ~~~show first piec -> soon: put it at the top left
    img = render_puzzle_piece(starting_piece, scale=0.5, margin=50)
    cv2.imshow("Firt Piece", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find next matching puzzle piece
    index, possible_matches = _find_matching_puzzle_piece(
        starting_piece, remaining_pieces
    )

    next_piece = None
    # show next piece
    if len(possible_matches) == 1:
        next_index, next_piece = possible_matches.popitem()

    img = render_puzzle_piece(next_piece, scale=0.5, margin=50)
    cv2.imshow("Second Piece", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _find_matching_puzzle_piece(
    origin: PuzzlePiece, remaining: dict[int, PuzzlePiece]
) -> tuple[int, dict[int, PuzzlePiece]]:
    prev_remaining: dict[int, PuzzlePiece] = remaining.copy()
    curr_remaining: dict[int, PuzzlePiece] = remaining.copy()

    o_length: int = len(origin.polygon.vertices)

    for i in range(0, o_length):
        o_points = origin.get_triplet(i, True)
        for j, r in prev_remaining.items():
            low, high = r.get_limits()
            low_index = low + i
            high_index = high - i
            if i == 0:
                # normal
                r_points_normal = r.get_triplet(low_index, True)
                if _first_segment_matches(o_points, r_points_normal):
                    continue

                # reverse
                r_points_reverse = r.get_triplet(high_index, False)
                if _first_segment_matches(o_points, r_points_reverse):
                    continue

                del curr_remaining[j]
            else:
                # normal
                r_points_normal = r.get_triplet(low_index, True)
                if _next_segment_matches(o_points, r_points_normal):
                    continue

                # reverse
                r_points_reverse = r.get_triplet(high_index, False)
                if _next_segment_matches(o_points, r_points_reverse):
                    continue

                del curr_remaining[j]

        if len(curr_remaining.items()) == 0:
            # in this round all the pieces drop out
            return (i - 1, prev_remaining)

        # there are still pieces in the pool
        prev_remaining = curr_remaining.copy()

    # if this is ever used, mistakes were made
    return (0, remaining)


def _first_segment_matches(
    origin: tuple[Point, Point, Point], match: tuple[Point, Point, Point]
) -> bool:
    return _length_matching((origin[1], origin[2]), (match[1], match[2]))


def _next_segment_matches(
    origin: tuple[Point, Point, Point], match: tuple[Point, Point, Point]
) -> bool:
    return _first_segment_matches(origin, match) and _angle_matching(origin, match)


def _length_matching(origin: tuple[Point, Point], match: tuple[Point, Point]) -> bool:
    o_this, o_next = origin
    m_this, m_next = match

    o_length: float = ((o_this.x - o_next.x) ** 2 + (o_this.y - o_next.y) ** 2) ** 0.5
    m_length: float = ((m_this.x - m_next.x) ** 2 + (m_this.y - m_next.y) ** 2) ** 0.5

    err_margin_len: float = o_length * ERROR_MARCHING_LENGTH

    return abs(o_length - m_length) <= err_margin_len


def _angle_matching(
    origin: tuple[Point, Point, Point], match: tuple[Point, Point, Point]
) -> bool:
    o_prev, o_this, o_next = origin
    m_prev, m_this, m_next = match

    o_angle = _angle_at_this(o_prev, o_this, o_next)
    m_angle = _angle_at_this(m_prev, m_this, m_next)

    return abs(o_angle - m_angle) <= ERROR_MARCHING_ANGLE / 2.0


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
