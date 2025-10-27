"""Code-File providing the matching of puzzle edges."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
    Edge2: Edge = pieces[3].get_edges["Top"]

    value = Edge1.compute_similarity(Edge2)

    print(value)

    
    x1, y1 = Edge1.get_plotvalues()

    y2 = Edge2.get_signature
    # y3 = [25, 20, 15, 10, 5, 0]

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
                temp_x, temp_y = temp_edge.get_plotvalues()
                plt.plot(temp_x, temp_y, label=f"Piece {i} - {name}", alpha=0.5)

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
