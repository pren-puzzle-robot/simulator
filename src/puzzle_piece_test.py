# load_corners_from_json.py

from __future__ import annotations

import json
import cv2
from pathlib import Path
from typing import Dict, List

from component.point import Point
from component.puzzle_piece import PuzzlePiece, PieceType
from component.draw_puzzle_piece import render_puzzle_piece


def _points_from_list(raw_points: List[List[float]]) -> List[Point]:
    """
    Convert a list like [[x1, y1], [x2, y2], ...] into a list of Point objects.
    """
    points: List[Point] = []
    for idx, pair in enumerate(raw_points):
        if len(pair) != 2:
            raise ValueError(f"Point at index {idx} does not have length 2: {pair!r}")
        x, y = pair
        points.append(Point(float(x), float(y)))
    return points


def load_corner_pieces(json_path: Path) -> Dict[str, PuzzlePiece]:
    """
    Load corner puzzle pieces from a JSON file with structure:
        {
          "piece_1.png": [[x1, y1], [x2, y2], ...],
          "piece_2.png": [[x1, y1], [x2, y2], ...],
          ...
        }

    Returns a dict mapping filename -> PuzzlePiece.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object with filename keys")

    pieces: Dict[str, PuzzlePiece] = {}

    for filename, raw_points in data.items():
        if not isinstance(raw_points, list):
            raise ValueError(f"Value for {filename!r} is not a list of points")

        points = _points_from_list(raw_points)

        # You can enforce at least 3 points if your Polygon requires it:
        if len(points) < 3:
            raise ValueError(
                f"Piece {filename!r} has fewer than 3 points ({len(points)}); "
                "cannot build a valid polygon"
            )

        piece = PuzzlePiece(points)
        pieces[filename] = piece

    return pieces


def main() -> None:
    json_path = "..\output\corners.json"

    pieces = load_corner_pieces(json_path)

    print(f"Loaded {len(pieces)} corner pieces from {json_path}")
    for name, piece in pieces.items():
        print(f"{name}: type={piece.type}, vertices={len(piece.polygon.vertices)}, area={piece.polygon.area():.2f}, perimeter={piece.polygon.perimeter():.2f}, centroid={piece.polygon.centroid()}")
        img = render_puzzle_piece(piece, scale=0.5, margin=50)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
