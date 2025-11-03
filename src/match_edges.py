"""Code-File providing the matching of puzzle edges."""

from pathlib import Path
import matplotlib.pyplot as plt

from component import PuzzlePiece, Edge


def main():
    """ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob", default="piece_*_edges.json", help="file glob for numeric edges"
    )
    ap.add_argument("--out", default="out", help="output folder")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print('No files matched. Example: --glob "piece_*_edges.json"')
        return

    with open("piece_1_edges.json", "r", encoding="utf-8") as file:
        numeric_data = json.load(file)

    signatureTop: List[float] = numeric_data["sides"]["Top"]["signature"]
    signatureLeft: List[float] = numeric_data["sides"]["Left"]["signature"]

    print("signatureTop", len(signatureTop))
    print("signatureLeft", len(signatureLeft))"""

    path = Path(__file__).parent.parent / "output" / "piece_1_edges.json"
    path.resolve()
    # print(path)

    piece1 = PuzzlePiece.from_json(path)
    # print(piece1)

    pieces: dict[int, PuzzlePiece] = {}

    for i in range(1, 5):
        path = Path(__file__).parent.parent / "output" / f"piece_{i}_edges.json"
        piece = PuzzlePiece.from_json(path)
        pieces[i] = piece

    value: float = float("inf")
    names: list[str] = ["Top", "Right", "Bottom", "Left"]

    Edge1: Edge = pieces[1].get_edges["Top"]
    Edge2: Edge = pieces[3].get_edges["Left"]

    # value = Edge1.compute_similarity(Edge2)

    valuez = Edge1.compute_difference(Edge2)

    # print(value)
    print(valuez)

    x1, y1 = Edge1.get_plotvalues()

    x2, y2 = Edge2.get_plotvalues()
    # y3 = [25, 20, 15, 10, 5, 0]

    # print(Edge1.get_local_middle_most_extrema())
    # print(Edge2.get_local_middle_most_extrema())

    """plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, label="Edge 1 Base", linewidth=2, alpha=0.5)
    plt.plot(x2, y2, label="Edge 2 Base", linewidth=2, alpha=0.5)

    value_offset = Edge1._get_off_set_between_signatures(Edge2)

    plt.plot(
        [a + value_offset[0] for a in x1],
        [b + value_offset[1] for b in y1],
        label="Edge 1 Alter",
        linewidth=2,
    )
    plt.plot(
        x2[0:-1],
        y2[0:-1],
        label="Edge 2 Alter",
        linewidth=2,
    )

    """
    # Plot erstellen
    plt.figure(figsize=(8, 5))  # Größe des Plots (optional)

    # Mehrere Linien plotten
    plt.plot(x1, y1, label="Base", linewidth=2)
    # plt.plot(x, y3, label="Abnehmend", linestyle=":")

    for i in range(1, 5):
        if i != 1:
            temp_piece: PuzzlePiece = pieces[i]
            for name in names:
                temp_edge: Edge = temp_piece.get_edges[name]
                temp_x, temp_y_values = Edge1._compute_matching_plots(temp_edge)
                temp_value: float = Edge1.compute_difference(temp_edge)
                plt.plot(
                    temp_x,
                    temp_y_values[1],
                    label=f"Piece {i} - {name} - {temp_edge.get_cat} - {temp_value}",
                    alpha=0.5,
                )

    # Achsenbeschriftungen und Titel
    plt.xlabel("x-Werte")
    plt.ylabel("y-Werte")
    plt.title("Mehrere Datenreihen in einem Plot")

    # Legende aktivieren
    plt.legend()

    # Gitter aktivieren (optional)
    plt.grid(True)

    # Plot als PNG speichern
    plt.savefig("mehrere_datenreihen.png", dpi=300, bbox_inches="tight")

    # Optional: Plot anzeigen
    plt.show()


if __name__ == "__main__":
    main()
