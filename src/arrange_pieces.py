import json, math, os
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

TOL_PIX = 3.5  # orientation tolerance

def orient_of(p1, p2, tol=TOL_PIX):
    dx = abs(p1[0] - p2[0]); dy = abs(p1[1] - p2[1])
    if dy <= tol and dx > dy: return "H"
    if dx <= tol and dy > dx: return "V"
    return "OTHER"

def load_inputs(corners_path: str, outer_edges_path: str):
    with open(corners_path, "r", encoding="utf-8") as f:
        corners = json.load(f)
    with open(outer_edges_path, "r", encoding="utf-8") as f:
        edges = json.load(f)
    return corners, edges

def classify_outer_edges(oe_list):
    out = []
    for e in oe_list:
        p1 = tuple(e["p1"]); p2 = tuple(e["p2"])
        out.append({
            "p1": p1, "p2": p2, "length": float(e["length"]),
            "indices": tuple(e["indices"]),
            "orientation": orient_of(p1, p2)
        })
    return out

def get_hlen(piece_oe):
    hs = [e for e in piece_oe if e["orientation"] == "H"]
    if hs: return hs[0]["length"]
    # fallback: most horizontal-ish segment
    es = sorted(piece_oe, key=lambda e: abs(e["p1"][0]-e["p2"][0]), reverse=True)
    return es[0]["length"]

def get_vlen(piece_oe):
    vs = [e for e in piece_oe if e["orientation"] == "V"]
    if vs: return vs[0]["length"]
    # fallback: most vertical-ish segment
    es = sorted(piece_oe, key=lambda e: abs(e["p1"][1]-e["p2"][1]), reverse=True)
    return es[0]["length"]

def common_endpoint(a1, a2, b1, b2):
    # nearest pair mid-point, in case floats differ slightly
    pairs = [(a1,b1),(a1,b2),(a2,b1),(a2,b2)]
    best = min(pairs, key=lambda p: (p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2)
    return ((best[0][0]+best[1][0])/2.0, (best[0][1]+best[1][1])/2.0)

def corner_anchor(oe):
    # pick a horizontal-like and a vertical-like edge; if missing, pick the most orthogonal pair
    hs = [e for e in oe if e["orientation"]=="H"]
    vs = [e for e in oe if e["orientation"]=="V"]
    if not hs or not vs:
        cand = oe[:]
        best, best_dot = (cand[0], cand[-1]), 1e18
        for i in range(len(cand)):
            for j in range(i+1, len(cand)):
                u = (cand[i]["p2"][0]-cand[i]["p1"][0], cand[i]["p2"][1]-cand[i]["p1"][1])
                v = (cand[j]["p2"][0]-cand[j]["p1"][0], cand[j]["p2"][1]-cand[j]["p1"][1])
                dot = abs(u[0]*v[0] + u[1]*v[1])
                if dot < best_dot:
                    best_dot = dot; best = (cand[i], cand[j])
        hs, vs = [best[0]], [best[1]]
    h, v = hs[0], vs[0]
    return common_endpoint(h["p1"], h["p2"], v["p1"], v["p2"])

def label_corners_by_geometry(corner_map: Dict[str, dict]):
    # Use anchor y/x to split top/bottom and left/right.
    # y grows downward; top = smaller y.
    items = [(name, data["anchor"]) for name, data in corner_map.items()]
    # sort by y
    tops = sorted(items, key=lambda it: it[1][1])[:2]
    bots = sorted(items, key=lambda it: it[1][1])[-2:]
    TL = min(tops, key=lambda it: it[1][0])[0]
    TR = max(tops, key=lambda it: it[1][0])[0]
    BL = min(bots, key=lambda it: it[1][0])[0]
    BR = max(bots, key=lambda it: it[1][0])[0]
    return TL, TR, BL, BR

def build_bands(pieces, meta, TL, TR, BL, BR):
    # bands are lists of (name, hlen/vlen)
    def hlen(n): return get_hlen(pieces[n]["oe"])
    def vlen(n): return get_vlen(pieces[n]["oe"])

    # Start with corners only
    top = [(TL, hlen(TL)), (TR, hlen(TR))]
    bottom = [(BL, hlen(BL)), (BR, hlen(BR))]
    left = [(TL, vlen(TL)), (BL, vlen(BL))]
    right = [(TR, vlen(TR)), (BR, vlen(BR))]

    # If any "edge" pieces exist, add them to the appropriate band by orientation and y/x proximity.
    for name, p in pieces.items():
        if meta[name]["type"] != "edge": continue
        oe = p["oe"]
        Hs = [e for e in oe if e["orientation"]=="H"]
        Vs = [e for e in oe if e["orientation"]=="V"]
        if Hs:
            # decide top vs bottom by edge y (closer to TL/TR y or BL/BR y)
            yH = (Hs[0]["p1"][1] + Hs[0]["p2"][1]) / 2.0
            y_top = (pieces[TL]["anchor"][1] + pieces[TR]["anchor"][1]) / 2.0
            y_bot = (pieces[BL]["anchor"][1] + pieces[BR]["anchor"][1]) / 2.0
            if abs(yH - y_top) <= abs(yH - y_bot):
                top.insert(1, (name, get_hlen(oe)))
            else:
                bottom.insert(1, (name, get_hlen(oe)))
        elif Vs:
            # decide left vs right by edge x
            xV = (Vs[0]["p1"][0] + Vs[0]["p2"][0]) / 2.0
            x_left = (pieces[TL]["anchor"][0] + pieces[BL]["anchor"][0]) / 2.0
            x_right = (pieces[TR]["anchor"][0] + pieces[BR]["anchor"][0]) / 2.0
            if abs(xV - x_left) <= abs(xV - x_right):
                left.insert(1, (name, get_vlen(oe)))
            else:
                right.insert(1, (name, get_vlen(oe)))

    return top, bottom, left, right

def best_fit_a5(top, bottom, left, right):
    # Sum lengths
    T = sum(h for _, h in top)
    B = sum(h for _, h in bottom)
    L = sum(v for _, v in left)
    R = sum(v for _, v in right)
    rt2 = math.sqrt(2.0)
    H_best = (rt2*(T + B) + (L + R)) / 6.0
    W_best = rt2 * H_best
    return W_best, H_best, T, B, L, R

def compute_translations(pieces, TL, TR, BL, BR, bands, W, H):
    top, bottom, left, right = bands
    # place corners to the four corners of the canvas
    placements = {}
    placements[TL] = {"side":"top-left", "anchor": pieces[TL]["anchor"], "target": (0.0, 0.0)}
    placements[TR] = {"side":"top-right", "anchor": pieces[TR]["anchor"], "target": (W, 0.0)}
    placements[BL] = {"side":"bottom-left", "anchor": pieces[BL]["anchor"], "target": (0.0, H)}
    placements[BR] = {"side":"bottom-right", "anchor": pieces[BR]["anchor"], "target": (W, H)}

    # helpers
    def left_offset(band):  # sum of previous h-lengths
        acc = {}
        x = 0.0
        for name, h in band:
            acc[name] = x
            x += h
        return acc

    # horizontal bands
    off_top = left_offset(top)
    off_bottom = left_offset(bottom)
    for name, _h in top:
        if name in placements: continue
        # anchor: left end of its horizontal outer edge
        e = next(e for e in pieces[name]["oe"] if orient_of(e["p1"], e["p2"])=="H")
        left_pt = e["p1"] if e["p1"][0] <= e["p2"][0] else e["p2"]
        placements[name] = {"side":"top", "anchor": left_pt, "target": (off_top[name], 0.0)}
    for name, _h in bottom:
        if name in placements: continue
        e = next(e for e in pieces[name]["oe"] if orient_of(e["p1"], e["p2"])=="H")
        left_pt = e["p1"] if e["p1"][0] <= e["p2"][0] else e["p2"]
        placements[name] = {"side":"bottom", "anchor": left_pt, "target": (off_bottom[name], H)}

    # vertical bands (not strictly needed to translate; corners already pin H, but we keep for completeness)
    # could be used to sanity-check

    # convert to tx,ty
    out = {}
    for name, plc in placements.items():
        ax, ay = plc["anchor"]; tx, ty = plc["target"]
        out[name] = {"tx": tx-ax, "ty": ty-ay, "side": plc["side"], "anchor": plc["anchor"], "target": plc["target"]}
    return out

def place(corners_path, outer_edges_path, out_layout="puzzle_layout.json", preview_png=None):
    polys, meta = load_inputs(corners_path, outer_edges_path)

    # Build piece dict
    pieces = {}
    corners_only = {}
    for name, poly in polys.items():
        oe = classify_outer_edges(meta[name]["outer_edges"])
        pieces[name] = {"poly": [tuple(p) for p in poly], "oe": oe, "type": meta[name]["type"]}

    # Collect corners (must be 4)
    corner_names = [n for n,p in pieces.items() if p["type"]=="corner"]
    if len(corner_names) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corner_names)}")

    # Corner anchors
    for n in corner_names:
        pieces[n]["anchor"] = corner_anchor(pieces[n]["oe"])

    # Order corners by image geometry
    TL, TR, BL, BR = label_corners_by_geometry({n: pieces[n] for n in corner_names})

    # Build bands (works for 4 or 6 total pieces)
    top, bottom, left, right = build_bands(pieces, meta, TL, TR, BL, BR)

    # Best-fit A5 landscape
    W, H, T, B, L, R = best_fit_a5(top, bottom, left, right)

    # Translations
    placements = compute_translations(pieces, TL, TR, BL, BR, (top, bottom, left, right), W, H)

    # Save layout
    os.makedirs(os.path.dirname(out_layout) or ".", exist_ok=True)
    payload = {
        "canvas_size": {"width": W, "height": H, "aspect": W/H, "A5": math.sqrt(2.0)},
        "band_sums": {"top": T, "bottom": B, "left": L, "right": R},
        "bands": {
            "top": [n for n,_ in top],
            "bottom": [n for n,_ in bottom],
            "left": [n for n,_ in left],
            "right": [n for n,_ in right],
        },
        "placements": placements
    }
    with open(out_layout, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Optional preview using polygons only
    if preview_png:
        from math import ceil
        Wc, Hc = ceil(W), ceil(H)
        img = Image.new("RGBA", (max(1,Wc), max(1,Hc)), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        for name, p in pieces.items():
            if name not in placements: continue
            dx, dy = placements[name]["tx"], placements[name]["ty"]
            poly = [(x+dx, y+dy) for (x,y) in p["poly"]]
            draw.polygon(poly, outline=(0,0,0,255))
        img.save(preview_png)

    return payload

if __name__ == "__main__":
    # Example:
    # python puzzle_place.py
    layout = place("../output/corners.json", "../output/outer_edges.json",
                   out_layout="../output/puzzle_layout.json",
                   preview_png="../output/puzzle_preview.png")
    print(json.dumps(layout, indent=2))
