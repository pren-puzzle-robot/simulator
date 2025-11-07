"""Code-File providing the matching of puzzle edges."""

import os

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from component import PuzzlePiece, Edge

DIRECTIONS = ["Top", "Right", "Bottom", "Left"]
COLOURS = ["blue", "orange", "green", "red"]
FILENAME = os.path.basename(__file__)
LIVE_DEMO = False  # set to True to show plots interactively

PUZZLE: dict[int, PuzzlePiece] = {}


def main():
    """main function to match edges and create plots"""

    _setup_puzzle_data()

    n: int = PUZZLE.__len__()

    for i in range(1, n + 1):
        _compute_plots_per_piece(i)

    manager = plt.get_current_fig_manager()
    manager.set_window_title(f"{FILENAME} - detail - P.1_Top & P.3_Top")

    base: Edge = PUZZLE[1].get_edges["Top"]
    other: Edge = PUZZLE[3].get_edges["Top"]

    x_base, y_base = base.get_plotvalues()
    x_other, y_other = other.get_plotvalues()

    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(x_base, y_base, color="blue", label="Piece 1 - Top - original", linewidth=2)
    plt.plot(x_other, y_other, color="orange", label="Piece 3 - Top - original", linewidth=2)


    plt.title("Kantenvergleich im Detail")
    plt.xlabel("Kantenlänge [px.]")
    plt.ylabel("d/dx des Höhenprofils [f'(px.)]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("../output/demo_detail_matching.png")


def _compute_ranking_per_edge(base: Edge, piece: int) -> dict[Edge, float]:
    edges: dict[Edge, float] = {}

    for i in range(1, 5):
        if i != piece:
            temp_piece: PuzzlePiece = PUZZLE[i]
            for name in DIRECTIONS:
                temp_edge: Edge = temp_piece.get_edges[name]
                temp_value: float = base.compute_similarity(temp_edge)
                edges[temp_edge] = temp_value

    sorted_edges = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

    return sorted_edges


def _compute_plots_per_piece(piece_id: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=100)

    n: int = PUZZLE.__len__()

    manager = plt.get_current_fig_manager()
    manager.set_window_title(f"{FILENAME} - output- ({piece_id}/{n})")

    fig.suptitle(f"Ergebnis der Kantenvergleiche - Puzzleteil {piece_id}", fontsize=16)

    piece: PuzzlePiece = PUZZLE[piece_id]

    for i in range(2):
        for j in range(2):
            temp_dir = DIRECTIONS[_combine_bits(i, j)]
            temp_colour = COLOURS[_combine_bits(i, j)]
            temp_subplot: Axes = axes[i, j]
            temp_edge: Edge = piece.get_edges[temp_dir]
            temp_ranking: dict[Edge, float] = _compute_ranking_per_edge(
                temp_edge, piece_id
            )
            _draw_subplots(temp_subplot, temp_edge, temp_colour, temp_ranking)

    plt.tight_layout()

    if LIVE_DEMO:
        plt.show()

    plt.savefig(f"../output/piece_{piece_id}_matching.png")


def _draw_subplots(
    subplot: Axes, base: Edge, colour: str, others: dict[Edge, float]
) -> None:
    x_base, y_base = base.get_plotvalues()
    subplot.plot(
        x_base,
        y_base,
        color="black",
        label="Original",
        linewidth=2,
    )

    score: float = 0.0
    score_set: bool = False

    for edge, value in others.items():
        if not score_set:
            score = value
            score_set = True

        if value + 0.1 >= score:
            color = colour
            linewidth = 2
        else:
            color = "gray"
            linewidth = 1

        x, y = edge.get_plotvalues()

        subplot.plot(
            x,
            y,
            color=color,
            label=f"Piece {edge.get_piece} - {edge.get_direction[0]} - {(value * 100):.2f}%",
            alpha=value,
            linewidth=linewidth,
        )

    subplot.set_title(f"Kante - {base.get_direction.capitalize()}")
    subplot.set_xlabel("Kantenlänge [px.]")
    subplot.set_ylabel("d/dx des Höhenprofils [f'(px.)]")
    subplot.legend()
    subplot.grid(True)


def _setup_puzzle_data() -> None:
    """Setup the needed information about the puzzle by\n
    scanning the json files and creating a `dict` with\n
    `PuzzlePieces` as elements to access their data more\n
    easily."""
    i: int = 1
    path: Path = Path(__file__).parent.parent / "output" / f"piece_{i}_edges.json"
    path.resolve()

    while path.exists():
        piece = PuzzlePiece.from_json(path)
        PUZZLE[i] = piece

        i += 1
        path = Path(__file__).parent.parent / "output" / f"piece_{i}_edges.json"


def _combine_bits(a: int, b: int) -> int:
    return (a << 1) | b


if __name__ == "__main__":
    main()
