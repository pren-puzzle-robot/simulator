import json, os, sys, math
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

# ---------- Tunables ----------
LONG_EDGE_VS_MEDIAN = 1.8   # edge is "long" if length >= median*this
LONG_EDGE_VS_MAX    = 0.60  # ...and also >= this * (longest length)
CORNER_ANGLE_RANGE  = (60, 120)  # degrees for the angle between two connecting outer edges
# ------------------------------

def seg_len(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return float(math.hypot(dx, dy))

def seg_angle(p1, p2):
    # angle of segment in radians
    return math.atan2(p2[1]-p1[1], p2[0]-p1[0])

def angle_between(v1, v2):
    # v1, v2: np.array([x,y])
    num = float(np.dot(v1, v2))
    den = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    a = math.degrees(math.acos(np.clip(num/den, -1.0, 1.0)))
    return a

def edges_from_polygon(poly: List[Tuple[float, float]]):
    """Return list of edges with indices and lengths (wrap closed)."""
    n = len(poly)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        p1, p2 = poly[i], poly[j]
        edges.append({
            "i": i, "j": j,
            "p1": (float(p1[0]), float(p1[1])),
            "p2": (float(p2[0]), float(p2[1])),
            "length": seg_len(p1, p2)
        })
    return edges

def pick_outer_edges(edges):
    """Heuristic:
       - mark 'long' edges vs median and vs global max
       - if two long edges share a vertex and form ~right angle -> corner piece
       - else one longest long edge -> edge piece
       - else -> inner piece
    """
    if not edges:
        return {"type": "inner", "outer_edges": []}

    lengths = np.array([e["length"] for e in edges], dtype=float)
    Lmax = float(lengths.max())
    Lmed = float(np.median(lengths))

    # candidate long edges
    long_edges = [e for e in edges if (e["length"] >= max(LONG_EDGE_VS_MEDIAN*Lmed, LONG_EDGE_VS_MAX*Lmax))]

    # Try to find a corner (two long edges sharing a vertex with ~90 deg)
    best_pair = None
    best_sum = -1.0
    for a in long_edges:
        for b in long_edges:
            if a["i"] == b["i"]:
                continue
            # share a vertex?
            shared = None
            if a["i"] == b["j"]:
                shared = a["i"]
                pa = np.array(a["p2"]) - np.array(a["p1"])
                pb = np.array(b["p1"]) - np.array(b["p2"])
            elif a["j"] == b["i"]:
                shared = a["j"]
                pa = np.array(a["p1"]) - np.array(a["p2"])
                pb = np.array(b["p2"]) - np.array(b["p1"])
            elif a["i"] == b["i"]:
                shared = a["i"]
                pa = np.array(a["p2"]) - np.array(a["p1"])
                pb = np.array(b["p2"]) - np.array(b["p1"])
            elif a["j"] == b["j"]:
                shared = a["j"]
                pa = np.array(a["p1"]) - np.array(a["p2"])
                pb = np.array(b["p1"]) - np.array(b["p2"])
            if shared is None:
                continue
            ang = angle_between(pa, pb)
            if CORNER_ANGLE_RANGE[0] <= ang <= CORNER_ANGLE_RANGE[1]:
                s = a["length"] + b["length"]
                if s > best_sum:
                    best_sum = s
                    best_pair = (a, b)

    if best_pair is not None:
        return {
            "type": "corner",
            "outer_edges": [
                {"i": best_pair[0]["i"], "j": best_pair[0]["j"], "p1": best_pair[0]["p1"], "p2": best_pair[0]["p2"], "length": best_pair[0]["length"]},
                {"i": best_pair[1]["i"], "j": best_pair[1]["j"], "p1": best_pair[1]["p1"], "p2": best_pair[1]["p2"], "length": best_pair[1]["length"]},
            ]
        }

    # Otherwise single longest long edge -> edge piece
    if long_edges:
        e = max(long_edges, key=lambda x: x["length"])
        return {
            "type": "edge",
            "outer_edges": [
                {"i": e["i"], "j": e["j"], "p1": e["p1"], "p2": e["p2"], "length": e["length"]}
            ]
        }

    # No long edges -> inner
    return {"type": "inner", "outer_edges": []}

def draw_edges(image_path, polygon, chosen, out_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    # draw polygon outline (light)
    pts = np.array(polygon, dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts], True, (180, 180, 0), 1, cv2.LINE_AA)

    # draw chosen edges (thick)
    for e in chosen["outer_edges"]:
        p1 = tuple(int(round(v)) for v in e["p1"])
        p2 = tuple(int(round(v)) for v in e["p2"])
        cv2.line(img, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(img, p1, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
    # label
    label = chosen["type"].upper()
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)

def main():
    # Inputs: folder containing piece_*.png + a corners.json from previous step
    src = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("../output")
    corners_json = Path(sys.argv[2]) if len(sys.argv) >= 3 else src / "../output/corners.json"
    out_json = src / "outer_edges.json"

    if not corners_json.exists():
        print(f"Could not find {corners_json}")
        sys.exit(1)

    with open(corners_json, "r", encoding="utf-8") as f:
        corners_map: Dict[str, List[List[float]]] = json.load(f)

    summary = {}
    for fname, coords in sorted(corners_map.items()):
        # coords assumed ordered (clockwise) from the previous script
        poly = [(float(x), float(y)) for x, y in coords]
        edges = edges_from_polygon(poly)
        chosen = pick_outer_edges(edges)

        summary[fname] = {
            "type": chosen["type"],
            "outer_edges": [{
                "p1": e["p1"], "p2": e["p2"], "length": round(float(e["length"]), 3),
                "indices": [int(e["i"]), int(e["j"])]
            } for e in chosen["outer_edges"]]
        }

        # overlay image
        img_path = src / fname
        out_path = src / f"{Path(fname).stem}_outer.png"
        if img_path.exists():
            draw_edges(str(img_path), poly, chosen, str(out_path))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json} and per-piece overlays (*_outer.png)")

if __name__ == "__main__":
    main()
