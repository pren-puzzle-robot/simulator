"""Code-File providing the matching of puzzle edges."""

from __future__ import annotations

import math

from component import PuzzlePiece, Point
from utilities import load_pieces, print_whole_puzzle_image

ERROR_MARCHING_LENGTH: float = 0.05
ERROR_MARCHING_ANGLE: float = 10  # degrees

PUZZLE: dict[int, PuzzlePiece] = load_pieces()


def main() -> None:
    """greedy matching"""

    # img = render_puzzle_piece(PUZZLE[origin[0]], scale=0.5, margin=50)
    # cv2.imshow(f"{1}. Piece", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print([x.get_indices for x in PUZZLE[1].outer_edges])
    # print(PUZZLE[1].get_limits())

    result = solve_greedy_corner_matching()

    # i: int = 1

    # img = render_puzzle_piece(PUZZLE[origin[0]], scale=0.5, margin=50)
    # cv2.imshow(f"{i}. Piece", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if result is not None:
        print(result)
    else:
        print("ERROR")
        return None

    print_whole_puzzle_image(PUZZLE)

    first_piece_id, first_edge = result.pop(0)
    first_piece = PUZZLE[first_piece_id]
    rotate_first_corner(first_piece)
    first_piece.translate(first_piece.outer_edges[-1].p1, Point(0, 0))

    previous_piece = first_piece
    previous_edge = first_edge

    odd: bool = True

    for next_piece_id, next_edge in result:
        current_piece = PUZZLE[next_piece_id]
        if odd:
            rotation = get_angle(
                (
                    previous_piece.polygon.vertices[previous_edge[0]],
                    previous_piece.polygon.vertices[previous_edge[1]],
                ),
                (
                    current_piece.polygon.vertices[next_edge[0]],
                    current_piece.polygon.vertices[next_edge[1]],
                ),
            )
            current_piece.rotate(rotation)

            # print_whole_puzzle_image(PUZZLE)

            current_piece.translate(
                current_piece.polygon.vertices[next_edge[0]],
                previous_piece.polygon.vertices[previous_edge[0]],
            )

            # print_whole_puzzle_image(PUZZLE)
            odd = False
        else:
            odd = True

        previous_piece = current_piece
        previous_edge = next_edge

    print_whole_puzzle_image(PUZZLE)


def get_angle(this_edge: tuple[Point, Point], next_edge: tuple[Point, Point]) -> float:
    """Rotates a puzzle piece to fit the current puzzle piece."""
    # Get the last outer edge of the current puzzle piece
    # current_edge = puzzle_piece.outer_edges[-1]

    # Get the first outer edge of the new piece
    # new_edge = piece.outer_edges[0]

    # Calculate the angle of the current edge
    dx1 = this_edge[1].x - this_edge[0].x
    dy1 = this_edge[1].y - this_edge[0].y
    angle1 = math.atan2(dy1, dx1)

    # Calculate the angle of the new edge
    dx2 = next_edge[1].x - next_edge[0].x
    dy2 = next_edge[1].y - next_edge[0].y
    angle2 = math.atan2(dy2, dx2)

    # Calculate the rotation needed to align the new edge with the current edge
    rotation_needed = angle1 - angle2

    return rotation_needed


def rotate_first_corner(puzzle_piece: PuzzlePiece) -> None:
    """Rotates the first corner piece to point down horizontally."""

    # Rotates the polygon so that the last outer edge is at the bottom
    bottom_edge = puzzle_piece.outer_edges[-1]
    # Calculate the angle of the bottom edge
    dx = bottom_edge.p2.x - bottom_edge.p1.x
    dy = bottom_edge.p2.y - bottom_edge.p1.y
    angle = math.atan2(dy, dx)
    rotation_needed = -angle + math.pi / 2  # Rotate to point downwards
    # Rotate all points in the polygon
    puzzle_piece.rotate(rotation_needed)
    pass


def solve_greedy_corner_matching() -> list[tuple[int, tuple[int, int]]] | None:
    """Greedy Corner Matching Algorithm"""

    return _pick_first_puzzle_piece()


def _pick_first_puzzle_piece() -> list[tuple[int, tuple[int, int]]] | None:
    # get the first corner piece
    index: int = next(i for i, p in PUZZLE.items() if p.is_corner)
    direction: bool = True
    origin: tuple[int, bool] = (index, direction)
    remaining_edges: list[int] = [k for k in PUZZLE if k != index]

    return _find_next_matching_puzzle_piece(origin, remaining_edges)


def _find_next_matching_puzzle_piece(
    origin: tuple[int, bool], remaining: list[int]
) -> list[tuple[int, tuple[int, int]]] | None:
    # flip-flopping list of remaining matches
    prev_remaining: list[tuple[int, bool]] = [(n, True) for n in remaining] + [
        (n, False) for n in remaining
    ]
    curr_remaining: list[tuple[int, bool]] = prev_remaining.copy()

    # setup values for the original puzzlepiece
    origin_piece = PUZZLE[origin[0]]
    o_direction = origin[1]
    o_length: int = len(origin_piece.polygon.vertices)
    o_limits: tuple[int, int] = origin_piece.get_limits()
    o_start: int = o_limits[0] if o_direction else o_limits[1]
    o_dir: int = 1 if o_direction else -1

    # default solution to be overritten
    this_edge: tuple[int, int] = (o_start, o_start)
    next_edges: list[tuple[int, bool]] = []

    for i in range(0, o_length):
        # current points and edges for origin
        o_index = o_start + i * o_dir
        o_points = origin_piece.get_triplet(o_index, o_direction)
        for m_piece, m_direction in prev_remaining:
            # current points and edges for a potential match
            match = PUZZLE[m_piece]
            m_limits = match.get_limits()
            m_start = m_limits[0] if m_direction else m_limits[1]
            m_dir = 1 if m_direction else -1
            m_index = m_start + i * m_dir
            m_points = match.get_triplet(m_index, m_direction)

            # result of the matching
            matching = (
                _first_segment_matches(o_points, m_points)
                if i == 0
                else _next_segment_matches(o_points, m_points)
            )

            # remove from pool if it does not fit
            if not matching:
                curr_remaining.remove((m_piece, m_direction))

        if len(curr_remaining) == 0:
            # in this round the remaining pieces droped out
            o_last = o_start + (i - 1) * o_dir
            this_edge = (o_start, o_last)
            next_edges = prev_remaining
            break

        # shuffel remaining pieces into the upcoming pool
        prev_remaining = curr_remaining.copy()

    # how to handle recursion

    # Failure: not a single Segment could match
    if (
        next_edges == remaining and this_edge[1] == o_start - 1 * o_dir
    ):  # could not complete a single loop
        return None

    if (
        len(next_edges) > 1
    ):  # multiple solutions -> test each one and return the first that went all the way
        for next_piece, next_dir in next_edges:
            next_edge = (next_piece, not next_dir)
            next_remaining = remaining.copy()
            next_remaining.remove(next_piece)
            temp_result = _find_next_matching_puzzle_piece(next_edge, next_remaining)

            if temp_result is not None:
                this_match: tuple[int, tuple[int, int]] = (origin[0], this_edge)

                edge_length: int = abs(this_edge[1] - this_edge[0]) + 1

                next_puzzle_piece: PuzzlePiece = PUZZLE[next_piece]
                next_limits: tuple[int, int] = next_puzzle_piece.get_limits()
                next_start: int = next_limits[0] if next_dir else next_limits[1]
                next_direction: int = 1 if next_dir else -1
                next_last: int = next_start + (edge_length - 1) * next_direction

                next_match: tuple[int, tuple[int, int]] = (
                    next_piece,
                    (next_start, next_last),
                )

                temp_result.insert(0, next_match)
                temp_result.insert(0, this_match)
                return temp_result

    elif len(next_edges) == 1:  # one possible next piece, easy solution
        next_piece, next_dir = next_edges.pop()
        next_edge = (next_piece, not next_dir)
        next_remaining = remaining.copy()
        next_remaining.remove(next_piece)

        this_match = (origin[0], this_edge)

        edge_length = abs(this_edge[1] - this_edge[0]) + 1

        next_puzzle_piece = PUZZLE[next_piece]
        next_limits = next_puzzle_piece.get_limits()
        next_start = next_limits[0] if next_dir else next_limits[1]
        next_direction = 1 if next_dir else -1
        next_last = next_start + (edge_length - 1) * next_direction

        next_match = (next_piece, (next_start, next_last))

        if len(next_remaining) == 0:  # we succeeded
            return [this_match, next_match]

        temp_result = _find_next_matching_puzzle_piece(next_edge, next_remaining)

        if temp_result is not None:
            temp_result.insert(0, next_match)
            temp_result.insert(0, this_match)
            return temp_result

        return None

    return None


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

    return abs(o_angle - m_angle) <= math.radians(ERROR_MARCHING_ANGLE / 2.0)


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

    # radiant angle between the two vectors
    angle_rad = math.atan2(cross, dot)
    return angle_rad


if __name__ == "__main__":
    main()
