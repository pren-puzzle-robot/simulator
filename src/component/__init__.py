# -*- coding: utf-8 -*-
"""Module with all the essential Components"""

__copyright__ = "Copyright (c) 2025 HSLU PREN Team 13, HS25. All rights reserved."

# simplifies access to these classes
from .puzzle_piece import PuzzlePiece
from .edge import Edge, EdgeDir
from .corner import Corner
from .solution import Solution
from .polygon import Polygon
from .point import Point

__all__ = ["Edge", "EdgeDir", "Corner", "PuzzlePiece", "Solution", "Polygon", "Point"]