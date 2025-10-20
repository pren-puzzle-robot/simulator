# annotate_piece_masks.py
import argparse, glob, os, re
import cv2 as cv
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def natural_key(s):
    # sort piece_2 before piece_10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def detect_corners_on_edge(gray, mask, *, max_corners=5000, quality=0.01, mindist=3,
                           blocksize=3, use_harris=True, harris_k=0.04):
    """Corner detection constrained to the mask; bias toward the outer edge."""
    # keep only inside the piece
    g = gray.copy()
    g[mask == 0] = 0

    # emphasize boundary
    edge = cv.Canny(g, 60, 180)
    # light boost at edge pixels to steer the detector to edges
    g = cv.add(g, (edge > 0).astype(np.uint8) * 20)

    pts = cv.goodFeaturesToTrack(
        g, maxCorners=max_corners, qualityLevel=quality, minDistance=mindist,
        blockSize=blocksize, useHarrisDetector=use_harris, k=harris_k
    )
    if pts is None:
        return np.empty((0,1,2), dtype=np.float32)

    # subpixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    pts = cv.cornerSubPix(g, np.float32(pts).reshape(-1,1,2), (5,5), (-1,-1), criteria)
    return pts

def annotate_one(mask_path, outdir, args, src_img=None, accum_overlay=None):
    # read mask as grayscale
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Skip (cannot read): {mask_path}")
        return 0, accum_overlay

    # binarize in case of compression artifacts
    _, bw = cv.threshold(mask, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # find main contour (largest area)
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        print(f"No contour in {mask_path}")
        return 0, accum_overlay
    c = max(contours, key=cv.contourArea)

    # build a per-piece color canvas (3-channel) for saving
    canvas = cv.cvtColor(bw, cv.COLOR_GRAY2BGR)

    # draw outline
    cv.drawContours(canvas, [c], -1, (0,255,255), 2)

    # pick grayscale for corner detection
    # if we have the source image and same size, use it for better texture; else use the mask itself
    if src_img is not None and src_img.shape[:2] == mask.shape[:2]:
        gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    else:
        gray = bw.copy()

    corners = detect_corners_on_edge(
        gray, bw,
        max_corners=args.max_corners,
        quality=args.quality,
        mindist=args.mindist,
        blocksize=args.blocksize,
        use_harris=(not args.tomasi),
        harris_k=args.harrisk
    )

    # draw corners
    count = 0
    if corners is not None and len(corners) > 0:
        for p in corners.reshape(-1,2):
            cv.circle(canvas, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            if accum_overlay is not None:
                cv.circle(accum_overlay, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        count = len(corners)

    # save per-piece annotated image
    name = os.path.splitext(os.path.basename(mask_path))[0]
    out_path = os.path.join(outdir, f"{name}_annotated.png")
    cv.imwrite(out_path, canvas)
    print(f"{name}: {count} corners -> {out_path}")

    # also outline onto overlay if provided
    if accum_overlay is not None:
        cv.drawContours(accum_overlay, [c], -1, (0,255,255), 2)

    return count, accum_overlay

def main():
    ap = argparse.ArgumentParser(description="Annotate piece_x.png masks with outlines and all corners.")
    ap.add_argument("--indir", required=True, help="folder with piece_*.png masks (full-size)")
    ap.add_argument("--outdir", required=True, help="folder to save annotated images")
    ap.add_argument("--pattern", default="piece_*.png", help="glob pattern for masks")
    ap.add_argument("--source", help="optional original photo to also write a combined overlay")
    ap.add_argument("--overlay_name", default="annotated_on_source.png", help="filename for the combined overlay")
    ap.add_argument("--max_corners", type=int, default=5000)
    ap.add_argument("--quality", type=float, default=0.1)
    ap.add_argument("--mindist", type=float, default=3.0)
    ap.add_argument("--blocksize", type=int, default=3)
    ap.add_argument("--tomasi", action="store_true", help="use Shi-Tomasi instead of Harris")
    ap.add_argument("--harrisk", type=float, default=0.04)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)), key=natural_key)
    if not paths:
        raise SystemExit(f"No masks found with pattern: {os.path.join(args.indir, args.pattern)}")

    # optional source image to paint a combined overlay
    src = None
    overlay = None
    if args.source:
        src = cv.imread(args.source)
        if src is None:
            print(f"Warning: could not read source: {args.source}")
        else:
            overlay = src.copy()

    total = 0
    for p in paths:
        cnt, overlay = annotate_one(p, args.outdir, args, src_img=src, accum_overlay=overlay)
        total += cnt

    if overlay is not None:
        out_overlay = os.path.join(args.outdir, args.overlay_name)
        cv.imwrite(out_overlay, overlay)
        print(f"Wrote combined overlay on source: {out_overlay}")

    print(f"Done. Masks: {len(paths)} | Total corners: {total}")

if __name__ == "__main__":
    main()
