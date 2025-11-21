from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .puzzle_piece_old import PuzzlePiece
from .point import Point


def _compute_bounds(piece: PuzzlePiece) -> Tuple[float, float, float, float]:
    verts = piece.polygon.vertices
    xs = [p.x for p in verts]
    ys = [p.y for p in verts]
    return min(xs), max(xs), min(ys), max(ys)


def _to_img_coords(p: Point, xmin: float, ymin: float, scale: float, margin: int) -> Tuple[int, int]:
    x = int((p.x - xmin) * scale) + margin
    y = int((p.y - ymin) * scale) + margin
    return x, y


def render_puzzle_piece(
    piece: PuzzlePiece,
    scale: float = 1.0,
    margin: int = 40,
) -> np.ndarray:
    """
    Render the puzzle piece to a new OpenCV image.

    - Polygon outline: black
    - Outer edges: red
    - Points: small circles with index labels
    - Type text in the top left corner
    """
    xmin, xmax, ymin, ymax = _compute_bounds(piece)
    w = int((xmax - xmin) * scale) + 2 * margin
    h = int((ymax - ymin) * scale) + 2 * margin

    if w <= 0:
        w = 200
    if h <= 0:
        h = 200

    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    verts = piece.polygon.vertices
    n = len(verts)

    # ----- Draw full polygon outline (black) -----
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        x1, y1 = _to_img_coords(p1, xmin, ymin, scale, margin)
        x2, y2 = _to_img_coords(p2, xmin, ymin, scale, margin)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # ----- Highlight outer edges (red, thicker) -----
    for e in piece.outer_edges:
        # Assuming OuterEdge has p1, p2 as Points
        x1, y1 = _to_img_coords(e.p1, xmin, ymin, scale, margin)
        x2, y2 = _to_img_coords(e.p2, xmin, ymin, scale, margin)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # ----- Draw points and indices -----
    for idx, p in enumerate(verts):
        x, y = _to_img_coords(p, xmin, ymin, scale, margin)
        # point marker
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        # index label
        cv2.putText(
            img,
            str(idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # ----- Draw type text -----
    type_text = piece.type.value.upper()
    cv2.putText(
        img,
        type_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 128, 0),
        2,
        cv2.LINE_AA,
    )

    return img
