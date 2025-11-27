"""Code-File providing the matching of puzzle edges."""

from __future__ import annotations

import math
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
from matplotlib.pyplot import draw
import numpy as np

from component import PuzzlePiece, Point
from component.draw_puzzle_piece import render_and_show_puzzle_piece
from utilities import load_pieces
from utilities.puzzle_piece_loader import PuzzlePieceLoader


PUZZLE: dict[int, PuzzlePiece] = PuzzlePieceLoader.load_pieces()

def rotate_first_corner(puzzlePiece: PuzzlePiece) -> None:
    """Rotates the first corner piece to point down horizontally."""

    # Rotates the polygon so that the last outer edge is at the bottom
    bottom_edge = puzzlePiece.outer_edges[-1]
    # Calculate the angle of the bottom edge
    dx = bottom_edge.p2.x - bottom_edge.p1.x
    dy = bottom_edge.p2.y - bottom_edge.p1.y
    angle = math.atan2(dy, dx)
    rotation_needed = -angle + math.pi / 2  # Rotate to point downwards
    # Rotate all points in the polygon
    puzzlePiece.rotate(rotation_needed)
    pass


def main() -> None:
    first_corner = next(i for i, p in PUZZLE.items() if p.is_corner)
    
    # render_and_show_puzzle_piece(PUZZLE[first_corner])
    
    rotate_first_corner(PUZZLE[first_corner])

    # render_and_show_puzzle_piece(PUZZLE[first_corner])

    order = solve_greedily(first_corner, PUZZLE)
    print("Solved order:", order)
    print("Number of pieces:", len(order))
    print("Rotation of pieces (radians):", [PUZZLE[pid].rotation for pid in order])

    move_pieces_to_fit(order, PUZZLE)

    print("Outer edges after moving:", "\n\n".join(str(PUZZLE[pid].outer_edges) for pid in order))

    print_puzzle_image(PUZZLE)


def solve_greedily(start_id: int, pieces: dict[int, PuzzlePiece]) -> list[int]:
    """Greedy recursive solver: always picks the next piece with most matching points."""
    remaining = pieces.copy()
    current = remaining.pop(start_id)  # remove start from pool
    order = [start_id]

    def recurse(current_piece: PuzzlePiece,
                remaining_pieces: dict[int, PuzzlePiece],
                placed_order: list[int]) -> None:
        if not remaining_pieces:
            print("All pieces processed. Order:", placed_order)
            return

        best_pid: int | None = None
        best_piece: PuzzlePiece | None = None
        best_score = -1

        # Try all remaining pieces and pick the one with most matches
        for pid, piece in remaining_pieces.items():
            # rotate_to_fit can mutate the piece in-place, which is fine
            next_piece = rotate_to_fit(current_piece, piece)
            print(f"Rotated piece {pid} to fit.")

            score = get_amount_of_matching_points(current_piece, next_piece)
            print(f"Piece {pid} has {score} matching points with current piece.")

            if score > best_score:
                best_score = score
                best_pid = pid
                best_piece = next_piece
        print()

        # Use the best matching piece as the next current
        assert best_pid is not None and best_piece is not None, "No suitable next piece found!"

        # render_and_show_puzzle_piece(best_piece)

        # Remove chosen piece from remaining and recurse
        remaining_copy = dict(remaining_pieces)
        remaining_copy.pop(best_pid)

        placed_order.append(best_pid)
        recurse(best_piece, remaining_copy, placed_order)

    recurse(current, remaining, order)
    return order

def move_pieces_to_fit(order: list[int], pieces: dict[int, PuzzlePiece]) -> None:
    """Moves pieces in the order to form an A5 size image."""

    first_piece = pieces[order[0]]
    first_piece.translate(first_piece.outer_edges[-1].p1, Point(0, 0))

    for idx in range(1, len(order)):
        current_piece = pieces[order[idx - 1]]
        next_piece = pieces[order[idx]]

        # Get the last outer edge of the current piece
        current_edge = current_piece.outer_edges[-1]
        # Get the first outer edge of the next piece
        next_edge = next_piece.outer_edges[0]

        MARGIN = 10.0 # margin between pieces

        # direction vector along the current edge
        dx = current_edge.p2.x - current_edge.p1.x
        dy = current_edge.p2.y - current_edge.p1.y

        # normalize
        length = math.hypot(dx, dy)
        ux = dx / length
        uy = dy / length

        # target point = p2 + margin along the direction
        target = Point(
            current_edge.p2.x + ux * MARGIN,
            current_edge.p2.y + uy * MARGIN
        )

        next_piece.translate(next_edge.p1, target)


def print_puzzle_image(pieces: dict[int, PuzzlePiece]) -> None:
    """Renders and prints the full puzzle image from the pieces."""
    all_points = []
    for piece in pieces.values():
        all_points.extend(piece.polygon.vertices)

    # Determine bounding box
    max_x = max(p.x for p in all_points)
    max_y = max(p.y for p in all_points)

    width = int(math.ceil(max_x))
    height = int(math.ceil(max_y))

    # Transparent background
    img = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # Deterministic color per piece name
    def color_for_name(piece_id):
        # fixed seed based on name for stable colors
        rnd = random.Random(hash(piece_id) & 0xFFFFFFFF)
        r = rnd.randint(50, 230)
        g = rnd.randint(50, 230)
        b = rnd.randint(50, 230)
        return (r, g, b, 255)

    # Render each polygon onto the image
    for pid, piece in pieces.items():
        outline = color_for_name(pid)
        fill = (outline[0], outline[1], outline[2], 40)  # very light transparent fill

        # Filled polygon with colored border
        draw.polygon([(p.x, p.y) for p in piece.polygon.vertices], fill=fill, outline=outline)

        cx, cy = piece.polygon.centroid().x, piece.polygon.centroid().y
        r = 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 255))

        label = "Piece {}\nRotation: {:.2f}\nTranslation: ({:.2f}, {:.2f})".format(pid, piece.rotation, piece.translation[0], piece.translation[1])
        draw.text((piece.polygon.centroid().x, piece.polygon.centroid().y), text=label, font=ImageFont.load_default(size=30))

    img.show()
         


def get_amount_of_matching_points(current: PuzzlePiece, next_piece: PuzzlePiece) -> int:
    """Calculates the amount of matching points between two puzzle pieces."""
    current_piece_index = current.outer_edges[-1].j
    next_piece_index = next_piece.outer_edges[0].i

    max_steps = min(len(current.polygon.vertices), len(next_piece.polygon.vertices))
    steps = 0

    matching_points = 0

    def point_distance(current_index: int, next_index: int) -> float:
        current_point = current.polygon.vertices[current_index % len(current.polygon.vertices)]
        current_point_next = current.polygon.vertices[(current_index + 1) % len(current.polygon.vertices)]

        next_point = next_piece.polygon.vertices[next_index % len(next_piece.polygon.vertices)]
        next_point_next = next_piece.polygon.vertices[(next_index - 1) % len(next_piece.polygon.vertices)]

        # Check whether the direction from current_point to current_point_next is similar to the direction from next_point to next_point_next
        dx1 = current_point_next.x - current_point.x
        dy1 = current_point_next.y - current_point.y
        dx2 = next_point_next.x - next_point.x
        dy2 = next_point_next.y - next_point.y
        return math.hypot(dx1 - dx2, dy1 - dy2)

    # Check, whether the points match
    while steps < max_steps:
        distance = point_distance(current_piece_index, next_piece_index)

        if distance < 15.0:  # Threshold for matching points
            matching_points += 1
            current_piece_index += 1
            next_piece_index -= 1
        else:
            # check if any of the next few points match
            found_better = False
            for look_ahead in range(1, 4):
                distance_ahead_1 = point_distance(current_piece_index, next_piece_index - look_ahead)
                distance_ahead_2 = point_distance(current_piece_index + look_ahead, next_piece_index)
                if distance_ahead_1 < 15.0 or distance_ahead_2 < 15.0:
                    matching_points += 1
                    current_piece_index += look_ahead + 1
                    next_piece_index -= look_ahead + 1
                    found_better = True
                    break
            if not found_better:
                break

        steps += 1


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

