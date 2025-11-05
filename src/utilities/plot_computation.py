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

import numpy as np
from scipy.signal import find_peaks


def analyze_plot(plot, min_prominence=0.1, min_distance=10):
    """
    Analyze a single (x, y) plot and return a list of peak points.

    Parameters
    ----------
    plot : tuple[list[float], list[float]]
        A tuple (x_values, y_values)
    min_prominence : float
        Minimum height prominence to consider a point a peak
    min_distance : int
        Minimum distance (in number of samples) between peaks

    Returns
    -------
    list[dict]
        Each dict has {'x': float, 'y': float, 'index': int, 'prominence': float}
    """
    x, y = plot
    x = np.asarray(x)
    y = np.asarray(y)

    # Find peaks
    peaks, props = find_peaks(y, prominence=min_prominence, distance=min_distance)

    # Collect peak information
    points = [
        {
            "index": int(i),
            "x": float(x[i]),
            "y": float(y[i]),
            "prominence": float(props["prominences"][j]),
        }
        for j, i in enumerate(peaks)
    ]

    return points


# TODO: Fix everything (x_x)
