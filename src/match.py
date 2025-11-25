"""Code-File providing the matching of puzzle edges."""

from __future__ import annotations

import math
import cv2

from component import PuzzlePiece, Point
from component.draw_puzzle_piece import render_puzzle_piece
from utilities import load_pieces
from utilities.puzzle_piece_loader import PuzzlePieceLoader


PUZZLE: dict[int, PuzzlePiece] = PuzzlePieceLoader.load_pieces()

def rotate_first_corner(puzzlePiece: PuzzlePiece) -> None:
    """Rotates the first corner piece to a defined orientation."""

    # Rotates the polygon so that the last outer edge is at the bottom
    bottom_edge = puzzlePiece.outer_edges[-1]
    # Calculate the angle of the bottom edge
    dx = bottom_edge.p2.x - bottom_edge.p1.x
    dy = bottom_edge.p2.y - bottom_edge.p1.y
    angle = math.atan2(dy, dx)
    # Calculate the rotation needed to make it horizontal
    rotation_needed = -angle
    # Rotate all points in the polygon
    puzzlePiece.rotate(rotation_needed)
    pass


def main() -> None:
    first_corner = next(i for i, p in PUZZLE.items() if p.is_corner)
    
    img = render_puzzle_piece(PUZZLE[first_corner], scale=0.5, margin=50)
    cv2.imshow(f"{1}. Piece", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rotate_first_corner(PUZZLE[first_corner])

    img = render_puzzle_piece(PUZZLE[first_corner], scale=0.5, margin=50)
    cv2.imshow(f"{1}. Piece", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    solving = PUZZLE.copy()
    solving.pop(first_corner)

    current = PUZZLE[first_corner]

    for pid, piece in solving.items():
        next_piece = rotate_to_fit(current, piece)
        print(f"Rotated piece {pid} to fit.")


        matching_points = get_amount_of_matching_points(current, next_piece)
        print(f"Piece {pid} has {matching_points} matching points with current piece.")

        img = render_puzzle_piece(next_piece, scale=0.5, margin=50)
        cv2.imshow(f"{pid}. Piece", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_amount_of_matching_points(current, next_piece) -> int:
    """Calculates the amount of matching points between two puzzle pieces."""
    current_piece_index = 0
    next_piece_index = next_piece.outer_edges[0].i

    matching_points = 0

    # Check, whether the points match
    while True:
        current_point = current.polygon.vertices[current_piece_index]
        current_point_next = current.polygon.vertices[(current_piece_index + 1) % len(current.polygon.vertices)]

        next_point = next_piece.polygon.vertices[next_piece_index]
        next_point_next = next_piece.polygon.vertices[(next_piece_index - 1) % len(next_piece.polygon.vertices)]

        # Check whether the direction from current_point to current_point_next is similar to the direction from next_point to next_point_next
        dx1 = current_point_next.x - current_point.x
        dy1 = current_point_next.y - current_point.y
        dx2 = next_point_next.x - next_point.x
        dy2 = next_point_next.y - next_point.y
        distance = math.hypot(dx1 - dx2, dy1 - dy2)

        if distance < 15.0:  # Threshold for matching points
            matching_points += 1
            current_piece_index += 1
            next_piece_index -= 1
        else:
            break

        if current_piece_index >= len(current.polygon.vertices) or next_piece_index < 0:
            break

    return matching_points

def rotate_to_fit(puzzle_piece: PuzzlePiece, piece: PuzzlePiece) -> PuzzlePiece:
    """Rotates a puzzle piece to fit the current puzzle piece."""
    # Get the last outer edge of the current puzzle piece
    current_edge = puzzle_piece.outer_edges[-1]

    # Get the first outer edge of the new piece
    new_edge = piece.outer_edges[0]

    # Calculate the angle of the current edge
    dx1 = current_edge.p2.x - current_edge.p1.x
    dy1 = current_edge.p2.y - current_edge.p1.y
    angle1 = math.atan2(dy1, dx1)

    # Calculate the angle of the new edge
    dx2 = new_edge.p2.x - new_edge.p1.x
    dy2 = new_edge.p2.y - new_edge.p1.y
    angle2 = math.atan2(dy2, dx2)

    # Calculate the rotation needed to align the new edge with the current edge
    rotation_needed = angle1 - angle2
    piece.rotate(rotation_needed)
    return piece


if __name__ == "__main__":
    main()

