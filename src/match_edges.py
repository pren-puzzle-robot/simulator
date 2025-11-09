"""Code-File providing the matching of puzzle edges."""

import os

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from utilities.plot_computation import compute_offset


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
    plt.plot(
        x_base,
        y_base,
        color="blue",
        label="Piece 1 - Top - original",
        linewidth=2,
        alpha=0.5,
    )
    plt.plot(
        x_other,
        y_other,
        color="orange",
        label="Piece 3 - Top - original",
        linewidth=2,
        alpha=0.5,
    )

    x_b_ex, y_b_ex = Edge.get_local_middle_most_extrema(x_base, y_base)
    x_o_ex, y_o_ex = Edge.get_local_middle_most_extrema(x_other, y_other)

    plt.plot(
        [x_b_ex, x_o_ex],
        [y_b_ex, y_o_ex],
        color="red",
        linestyle="--",
        label="Extrema Abweichung",
    )

    x_other_adapted = list(a + abs(x_b_ex - x_o_ex) for a in x_other)
    y_other_adapted = list(b - abs(y_b_ex - y_o_ex) for b in y_other)

    x_final, y_values_final = Edge._compute_matching_plots(
        x_base, y_base, x_other_adapted, y_other_adapted
    )

    plt.plot(
        x_final,
        list(a for (a, b) in y_values_final),
        color="blue",
        label="Piece 1 - Top - ajusted",
        linewidth=2,
    )
    plt.plot(
        x_final,
        list(b for (a, b) in y_values_final),
        color="orange",
        label="Piece 3 - Top - ajusted",
        linewidth=2,
    )

    plt.title("Kantenvergleich im Detail")
    plt.xlabel("Kantenlänge [px.]")
    plt.ylabel("d/dx des Höhenprofils [f'(px.)]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig("../output/demo_detail_matching.png")

    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax1.set_title("Beispiel-Plot zur Peak-Analyse")
    ax1.set_xlabel("X-Achse [px.]")
    ax1.set_ylabel("Y-Achse [Höhe f'(px.)]")
    ax1.grid(True)

    # Example data: mostly flat, two peaks
    # x = list(range(100))
    # y = [0] * 20 + [1, 3, 6, 9, 6, 3, 1] + [0] * 30 + [0, 2, 4, 8, 4, 2, 0] + [0] * 36

    plot_a = PUZZLE[2].get_edges["Top"].get_plotvalues()
    plot_b = PUZZLE[4].get_edges["Left"].get_plotvalues()

    dx, dy = compute_offset(plot_a, plot_b)
    # peaks = analyze_plot(
    #    (x, y),
    #    min_prominence=2,
    #    min_distance=10,
    # )
    # print(peaks)

    # Optional: visualize
    ax1.plot(
        plot_a[0],
        plot_a[1],
        label="Piece 2 - Top",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        plot_b[0],
        plot_b[1],
        label="Piece 4 - Left",
        color="orange",
        linewidth=2,
        alpha=0.5,
    )
    ax1.plot(
        list(x + dx for x in plot_b[0]),
        list(y + dy for y in plot_b[1]),
        label="Piece 4 - Left (adjusted)",
        color="orange",
        linewidth=2,
    )
    # for x_p, y_p in peaks:
    #    ax1.plot(x_p, y_p, "ro")
    # fig1.show()
    ax1.legend()
    fig1.savefig("../output/demo_peak_analysis.png")


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

        if value * 1.05 >= score:
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
            alpha=max(0.0, value),
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
