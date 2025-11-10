import cv2
import numpy as np
import sys, os, json
from math import acos, degrees
from glob import glob
import re

def remove_collinear(pts, angle_thresh_deg=175):
    """Drop nearly collinear points (angle close to 180°)."""
    if len(pts) <= 3:
        return pts
    keep = []
    for i in range(len(pts)):
        p0 = pts[(i-1) % len(pts)][0]
        p1 = pts[i][0]
        p2 = pts[(i+1) % len(pts)][0]
        v1, v2 = p0 - p1, p2 - p1
        c = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
        ang = degrees(acos(np.clip(c, -1, 1)))
        if ang < angle_thresh_deg:
            keep.append([p1])
    return np.array(keep, dtype=np.int32)

def detect_corners(image_path, output_path="corners_output.png", approx_frac=0.002, angle_thresh=175):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Error: could not open {image_path}")
        return None

    # Threshold with Otsu
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours (outer boundary)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        print(f"⚠️ No contour found in {image_path}")
        return None

    cnt = max(cnts, key=cv2.contourArea)

    # Simplify contour
    eps = approx_frac * cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, eps, True)
    poly = remove_collinear(poly, angle_thresh_deg=angle_thresh)

    # Draw results
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color_img, [poly], -1, (0, 255, 255), 2)
    for i, p in enumerate(poly):
        x, y = p[0]
        cv2.circle(color_img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(color_img, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imwrite(output_path, color_img)
    print(f"✅ Saved {output_path} with {len(poly)} corners")

    # Return list of corners as (x, y)
    return [(int(p[0][0]), int(p[0][1])) for p in poly]

if __name__ == "__main__":
    # Default folder: current or specified
    src_folder = sys.argv[1] if len(sys.argv) >= 2 else "../output"
    output_json = os.path.join(src_folder, "corners.json")

    print(f"📁 Scanning folder: {src_folder}")
    images = [
        f for f in glob(os.path.join(src_folder, "piece_*.png"))
        if re.fullmatch(r".*piece_\d+\.png", f)
    ]
    results = {}

    if not images:
        print("⚠️ No piece_*.png files found.")
        sys.exit(1)

    for img_path in images:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(src_folder, f"{name}_output{ext}")
        corners = detect_corners(img_path, out_path)
        if corners is not None:
            results[filename] = corners

    # Save all results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved all corners to {output_json}")
