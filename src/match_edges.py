from pathlib import Path

from component import PuzzlePiece


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
    print(path)

    piece1 = PuzzlePiece.from_json(path)
    print(piece1)


if __name__ == "__main__":
    main()
