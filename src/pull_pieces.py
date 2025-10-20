# Pull out puzzle piece contours from an image and save individual masks.

# python .\pull_pieces.py --image ..\sample_images\simple_1_rotated.png --outdir ..\output
import argparse, json, os
import cv2 as cv
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def preprocess(img):
    # robust contrast + denoise to help thresholding
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv.merge([l,a,b])
    img_eq = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    blur = cv.GaussianBlur(img_eq, (5,5), 0)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    return gray

def segment_foreground(gray):
    # Otsu threshold + morphology to isolate pieces
    thr_val, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # If background is white, invert so pieces are white
    if np.mean(bw) > 127:
        bw = cv.bitwise_not(bw)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel, iterations=2)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel, iterations=2)
    # fill small holes
    cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    cv.drawContours(mask, cnts, -1, 255, thickness=cv.FILLED)
    return mask

def find_pieces(mask, min_area=2000):
    # label connected components via contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv.contourArea(c) >= min_area]
    return contours

def convexity_features(contour):
    # convex hull and defects for concave “holes”
    hull = cv.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return [], [], None
    defects = cv.convexityDefects(contour, hull)
    holes_pts = []
    tabs_pts = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i,0]
            far = tuple(contour[f][0])
            # filter shallow defects to reduce noise (depth is scaled by 256)
            if depth > 8*256:
                holes_pts.append(far)  # concave
    # Convex “tabs”: approximate by finding hull points with high curvature
    hull_pts = cv.convexHull(contour, returnPoints=True).reshape(-1,2)
    if len(hull_pts) >= 5:
        # compute angle at each hull vertex
        for i in range(len(hull_pts)):
            p_prev = hull_pts[(i-1)%len(hull_pts)]
            p = hull_pts[i]
            p_next = hull_pts[(i+1)%len(hull_pts)]
            v1 = p_prev - p
            v2 = p_next - p
            if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0: 
                continue
            cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            cosang = np.clip(cosang, -1.0, 1.0)
            angle = np.degrees(np.arccos(cosang))
            # “tab” corners tend to be sharper than flat edges; tune threshold as needed
            if angle < 110:  # heuristic
                tabs_pts.append(tuple(p))
    return tabs_pts, holes_pts, hull_pts

def save_contours_only(img, contours, outdir):
    summary = []
    for idx, c in enumerate(contours, start=1):
        area = cv.contourArea(c)
        perim = cv.arcLength(c, True)

        # save a binary mask of each contour
        piece_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.drawContours(piece_mask, [c], -1, 255, thickness=cv.FILLED)
        cv.imwrite(os.path.join(outdir, f"piece_{idx}.png"), piece_mask)

        summary.append({
            "piece_id": idx,
            "area_px": float(area),
            "perimeter_px": float(perim)
        })

    with open(os.path.join(outdir, "edges.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser(description="Detect puzzle piece edges and interlocks")
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--outdir", default="outputs", help="folder to save results")
    ap.add_argument("--min_area", type=int, default=2000, help="min contour area to keep")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    img = cv.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    gray = preprocess(img)
    fg = segment_foreground(gray)
    contours = find_pieces(fg, min_area=args.min_area)

    # optional: refine contours using Canny edges along the mask for crisper boundaries
    edges = cv.Canny(gray, 60, 180)
    edges = cv.bitwise_and(edges, edges, mask=fg)
    # (kept for inspection; not required) 
    cv.imwrite(os.path.join(args.outdir, "edges.png"), edges)

    summary = save_contours_only(img, contours, args.outdir)
    print(f"Saved {len(contours)} piece masks, annotated overlay, and edges.json to {args.outdir}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
