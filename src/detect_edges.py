# detect_edges.py
import argparse
import json
import os
import glob
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("no contours")
    cnt = max(cnts, key=cv2.contourArea)
    cnt = cnt[:, 0, :]  # (N,2)
    return cnt


def polygon_orientation(pts):
    # shoelace; >0 if CCW
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def min_area_rect_sides(cnt):
    # Rotated rectangle around the piece
    rect = cv2.minAreaRect(cnt.astype(np.float32))
    box = cv2.boxPoints(rect)  # 4x2 (float)
    # order points clockwise
    c = np.mean(box, axis=0)
    angles = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    order = np.argsort(angles)
    box = box[order]
    # Build sides as (p0, p1)
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    return np.array(box), sides  # box is clockwise


def project_point_to_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    return a + np.clip(t, 0.0, 1.0) * ab, np.clip(t, 0.0, 1.0)


def signed_distance_to_line(p, a, b, outward_normal):
    ab = b - a
    n = np.array([-ab[1], ab[0]], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    # ensure n points outward
    if np.dot(n, outward_normal) < 0:
        n = -n
    return np.dot(p - a, n)


def outward_normal_for_side(a, b, centroid):
    mid = (a + b) * 0.5
    ab = b - a
    n = np.array([-ab[1], ab[0]], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    # point outward = away from centroid
    if np.dot(n, centroid - mid) > 0:
        n = -n
    return n


def extract_edge_signature(cnt, a, b, centroid, samples=256):
    """
    Returns t in [0,1], signed offsets along the normal (positive=tab, negative=hole)
    for the contour points closest to this rectangle side.
    """
    # Assign each contour point to this side if it projects inside the segment
    proj_pts = []
    dists_signed = []
    ts = []

    outward_n = outward_normal_for_side(a, b, centroid)
    ab = b - a

    # distance threshold: points far from the side line (e.g., corners of other sides)
    # are excluded; we set a generous threshold as 10% of side length
    side_len = np.linalg.norm(ab)
    max_dist = 0.1 * side_len + 1.0

    for p in cnt:
        p2, t = project_point_to_segment(p.astype(np.float64), a, b)
        # unsigned distance to infinite line
        line_dist = np.linalg.norm(np.array([p2[0] - p[0], p2[1] - p[1]]))
        if line_dist <= max_dist and 0.0 <= t <= 1.0:
            sd = signed_distance_to_line(p.astype(np.float64), a, b, outward_n)
            proj_pts.append(p2)
            dists_signed.append(sd)
            ts.append(t)

    if len(ts) < 8:
        # fallback: sample the ideal straight side (flat)
        t = np.linspace(0, 1, samples)
        return t, np.zeros_like(t)

    ts = np.array(ts)
    dists_signed = np.array(dists_signed)

    # Build a fixed-length signature by averaging in bins across t
    t_grid = np.linspace(0, 1, samples)
    sig = np.zeros_like(t_grid)
    counts = np.zeros_like(t_grid)

    # map each sample to nearest bin
    idx = np.clip((ts * (samples - 1)).astype(int), 0, samples - 1)
    for i, sd in zip(idx, dists_signed):
        sig[i] += sd
        counts[i] += 1
    counts[counts == 0] = 1
    sig = sig / counts
    # light smoothing
    sig = gaussian_filter1d(sig, sigma=3)
    return t_grid, sig


def classify_signature(sig, side_len):
    # normalize by side length to be scale independent
    s = sig / (side_len + 1e-12)
    m, p2 = float(np.mean(s)), float(np.max(np.abs(s)))
    # decision by mean and amplitude
    if p2 < 0.01:
        cls = "flat"
    elif m > 0:
        cls = "tab"
    else:
        cls = "hole"
    return cls, {"mean": m, "amplitude": p2}


def visualize(mask, box, sides_info, out_png):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [box.astype(int)], 0, (0, 0, 255), 2)
    # draw side labels near midpoints
    for k, info in sides_info.items():
        a, b = info["segment"]
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        mid = (a + b) * 0.5
        txt = f"{k}: {info['class']}"
        cv2.putText(
            vis,
            txt,
            (int(mid[0]), int(mid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(out_png, vis)


def process_piece(path, out_dir, samples=256):
    name = os.path.splitext(os.path.basename(path))[0]
    mask = load_mask(path)
    cnt = largest_contour(mask)
    # ensure contour CCW for consistency
    if polygon_orientation(cnt) < 0:
        cnt = cnt[::-1]
    centroid = np.mean(cnt, axis=0)

    box, sides = min_area_rect_sides(cnt)
    print(f"[{name}] box corners: {box.tolist()}")
    print(f"[{name}] side segments: {sides}")
    sides_info = {}
    # label sides consistently based on rectangle orientation:
    # order is clockwise; we label them Top, Right, Bottom, Left
    labels = ["Top", "Right", "Bottom", "Left"]
    for i, (label, seg) in enumerate(zip(labels, sides)):
        a, b = seg
        t, sig = extract_edge_signature(cnt, a, b, centroid, samples=samples)
        side_len = np.linalg.norm(b - a)
        cls, stats = classify_signature(sig, side_len)
        sides_info[label] = {
            "class": cls,
            "stats": stats,
            "segment": (a.tolist(), b.tolist()),
            "signature": sig.tolist(),
        }

    # write JSON
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}_edges.json")
    with open(json_path, "w") as f:
        json.dump({"piece": name, "sides": sides_info}, f, indent=2)

    # visualization
    png_path = os.path.join(out_dir, f"{name}_viz.png")
    visualize(mask, box, sides_info, png_path)

    # quick Matplotlib plot of signatures
    plt.figure(figsize=(7, 3))
    for label in ["Top", "Right", "Bottom", "Left"]:
        sig = np.array(sides_info[label]["signature"])
        plt.plot(
            np.linspace(0, 1, len(sig)),
            sig,
            label=f"{label} ({sides_info[label]['class']})",
        )
    plt.xlabel("normalized position")
    plt.ylabel("signed offset (px)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{name}_signatures.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[{name}] -> {json_path}, {png_path}, {plot_path}")
    return json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="piece_*.png", help="file glob for pieces")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--samples", type=int, default=256, help="signature length")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print('No files matched. Example: --glob "piece_*.png"')
        return

    for path in files:
        try:
            process_piece(path, args.out, samples=args.samples)
        except Exception as e:
            print(f"Failed {path}: {e}")


if __name__ == "__main__":
    main()
