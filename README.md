# Puzzle Simulator – PREN1

## Overview

This project implements a **software-based simulator** for solving a 2D puzzle as part of the **PREN1** module.
The simulator serves as a **digital twin** of the later physical system and enables the development, testing, and validation of core algorithms **without hardware**.

## Goals

- Simulate the autonomous puzzle-solving process
- Develop and compare different matching strategies
- Reduce technical and integration risks before hardware development
- Provide a clear, testable, and extendable software architecture

## Functionality

The simulator follows a structured processing pipeline:

1. Image preprocessing and segmentation of individual puzzle pieces
2. Contour and corner detection for each piece
3. Geometric domain modeling (`PuzzlePiece`, `Polygon`, `Point`, `OuterEdge`)
4. Puzzle assembly using interchangeable solver strategies

## Requirements

- Python 3.9 or newer
- OpenCV
- NumPy

Installation example:

```powershell
pip install opencv-python numpy pillow
```

## Usage

Run the simulator from the project root:

```powershell
python src/simulator.py --image sample_image/example.png --variant match
```

### Parameters

- `--image` – Path to the input image
- `--variant` – Solver strategy (`match` or `greedy`)

## Output

- Debug images (segmented pieces, detected corners)
- Intermediate data (e.g. JSON files)
- Visualization of the final assembled puzzle

All outputs (Intermediate Images/Files) are written to the `output/` directory.

## Context

Developed as part of **PREN1** at Hochschule Luzern.
