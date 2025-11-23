# -*- coding: utf-8 -*-
"""Module with all the essential Components"""

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."

# simplifies access to these classes
from .point import Point
from .puzzle_piece_loader import PuzzlePieceLoader
from .plot_computation import compute_offset


def load_pieces():
    """
    Returns a dictionary with all the puzzle pieces\n
    found in output and adds their value as a key to\n
    find them more easily.
    """
    return PuzzlePieceLoader.load_pieces()


__all__ = ["Point", "load_pieces", "compute_offset"]
