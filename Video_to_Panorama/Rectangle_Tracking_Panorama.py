#!/usr/bin/env python3
"""
Shapes_Track_Panorama.py

Post-alignment rectangle tracking and panorama generation:

1) Reads a pre-aligned video
2) Detects rectangles per frame
3) Tracks motion using rectangle matching
4) Produces:
   - debug tracking preview
   - panorama strip image
   - metadata CSV
"""

import cv2
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
import csv

# =========================
# USER-TUNABLE PARAMETERS
# =========================
VIDEO_PATH = "train.mp4"

# Rectangle detection / tracking
DETECT_DOWNSCALE = 0.45
MIN_RECT_AREA = 250
MAX_MATCH_DIST = 80.0
MAX_POINT_DX = 120.0
MAX_ANGLE_DEG = 30.0

# Polygon approximation factor
RECT_APPROX_EPS = 0.07
MIN_ASPECT = 0.5
MAX_ASPECT = 10.0
MIN_SOLIDITY = 0.1

# Panorama sampling
SCAN_X = 0.5
BASE_STRIP_WIDTH = 35

# Preview / debug
SHOW_PREVIEW = True
PREVIEW_FPS = 300
DRAW_MATCHES = True
DRAW_RECTS = True

# GPU
FORCE_CPU = False

# EMA smoothing
EMA_ALPHA = 0.25

# Output files
DEBUG_VIDEO = "TEMP/debug_shapetrack.mp4"
OUTPUT_PANORAMA = "TEMP/train_panorama.jpg"
OUTPUT_META = "TEMP/panorama_meta.csv"

# =========================
# Utility: CUDA detection
def has_cuda():
    try:
        if FORCE_CPU:
            return False
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

USE_CUDA = has_cuda()
cuda_canny = None
if USE_CUDA:
    try:
        cuda_canny = cv2.cuda.createCannyEdgeDetector(50,150)
    except Exception:
        USE_CUDA = False
        cuda_canny = None

def edges_cpu(gray):
    return cv2.Canny(gray,80,200)
def edges_cuda(gray):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(gray)
    edges_gpu = cuda_canny.detect(gpu)
    return edges_gpu.download()

# -------------------------
# Rectangle detection
# -------------------------
def find_rectangles(frame, min_area, eps_factor=RECT_APPROX_EPS):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,5,75,75)
    if USE_CUDA and cuda_canny:
        edges = edges_cuda(gray)
    else:
        edges = edges_cpu(gray)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, eps_factor*peri, True)
        if len(approx) !=4 or not cv2.isContourConvex(approx):
            continue
        xs = approx[:,0,0]; ys = approx[:,0,1]
        x1,y1,x2,y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        w,h = x2-x1, y2-y1
        if h<=0 or w<=0:
            continue
        aspect = w/h
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0
        solidity = area/hull_area if hull_area>0 else 0
        if solidity<MIN_SOLIDITY:
            continue
        cx,cy = xs.mean(), ys.mean()
        rects.append({"centroid":(cx,cy),"w":w,"h":h,"area":area,"bbox":(x1,y1,x2,y2),"poly":approx})
    return rects

# -------------------------
# Match rectangles
# -------------------------
def match_rects(prev, cur, max_dist=MAX_MATCH_DIST):
    if len(prev)==0 or len(cur)==0:
        return [], list(range(len(prev))), list(range(len(cur)))
    prev_cent = np.array([p["centroid"] for p in prev])
    cur_cent = np.array([c["centroid"] for c in cur])
    dists = np.linalg.norm(prev_cent[:,None,:]-cur_cent[None,:,:],axis=2)
    matches=[]
    used_prev=set(); used_cur=set()
    INF=np.max(dists)+1.0
    while True:
        idx = np.unravel_index(np.argmin(dists,axis=None),dists.shape)
        p,c = idx
        if dists[p,c]==INF or dists[p,c]>max_dist:
            break
        matches.append((p,c))
        used_prev.add(p); used_cur.add(c)
        dists[p,:]=INF; dists[:,c]=INF
        if np.all(dists==INF):
            break
    unmatched_prev = [i for i in range(len(prev)) if i not in used_prev]
    unmatched_cur = [j for j in range(len(cur)) if j not in used_cur]
    return matches, unmatched_prev, unmatched_cur

# -------------------------
# Main pipeline
# -------------------------
def main():
    print("Shapes_Track_Panorama.py starting...")
    print(f"CUDA available: {USE_CUDA}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened {VIDEO_PATH} | frames={total_frames} | fps={fps:.2f}")

    # tracking downscale
    det_w = max(64, int(w * DETECT_DOWNSCALE))
    det_h = max(64, int(h * DETECT_DOWNSCALE))
    scale_x = w / det_w
    scale_y = h / det_h
    scan_x_px = int(w * SCAN_X)

    prev_rects = []
    panorama_strips = []
    meta = []
    ema_dx = 0.0
    first_measure = True
    cumulative_distance_full = 0.0

    pbar = tqdm(total=total_frames, desc="Track+Panorama")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # downscale for detection
        small = cv2.resize(frame, (det_w, det_h))
        rects = find_rectangles(small, MIN_RECT_AREA)

        # match rectangles
        matches, unmatched_prev, unmatched_cur = match_rects(prev_rects, rects)
        dxs = []
        for p, c in matches:
            cx_prev, cy_prev = prev_rects[p]["centroid"]
            cx_cur, cy_cur = rects[c]["centroid"]
            dx = (cx_cur - cx_prev) * scale_x
            dy = (cy_cur - cy_prev) * scale_y
            angle_deg = math.degrees(math.atan2(dy, dx))
            if abs(dx) > MAX_POINT_DX or abs(angle_deg) > MAX_ANGLE_DEG:
                continue
            dxs.append(dx)
            if SHOW_PREVIEW and DRAW_MATCHES:
                x0, y0 = int(cx_prev * scale_x), int(cy_prev * scale_y)
                x1, y1 = int(cx_cur * scale_x), int(cy_cur * scale_y)
                cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # EMA smoothing
        if len(dxs) > 0:
            med_dx = np.median(dxs)
            if first_measure:
                ema_dx = med_dx
                first_measure = False
            else:
                ema_dx = EMA_ALPHA * med_dx + (1 - EMA_ALPHA) * ema_dx
        else:
            med_dx = 0.0

        # Panorama strip sampling
        strip_w = BASE_STRIP_WIDTH
        strip_x = scan_x_px - strip_w // 2
        strip = frame[:, strip_x:strip_x + strip_w, :].copy()
        panorama_strips.append(strip)
        cumulative_distance_full += ema_dx

        # Preview
        if SHOW_PREVIEW:
            if DRAW_RECTS:
                for r in rects:
                    x1, y1, x2, y2 = r["bbox"]
                    cv2.rectangle(frame, (int(x1 * scale_x), int(y1 * scale_y)),
                                  (int(x2 * scale_x), int(y2 * scale_y)), (255, 0, 0), 1)
            cv2.imshow("Tracking Preview", frame)
            if cv2.waitKey(int(1000 / PREVIEW_FPS)) & 0xFF == ord('q'):
                break

        # Metadata
        meta.append({
            "frame": frame_idx,
            "scan_x": scan_x_px,
            "strip_w": strip_w,
            "med_dx_full": med_dx,
            "near_shapes": len(rects)
        })

        prev_rects = rects
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    # build panorama
    if len(panorama_strips) > 0:
        panorama = np.hstack(panorama_strips)
        cv2.imwrite(OUTPUT_PANORAMA, panorama)
        print(f"Saved panorama: {Path(OUTPUT_PANORAMA).resolve()} width={panorama.shape[1]}")

    # write metadata
    if len(meta) > 0:
        with open(OUTPUT_META, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
            writer.writeheader()
            writer.writerows(meta)
        print(f"Saved metadata CSV: {Path(OUTPUT_META).resolve()}")

if __name__=="__main__":
    main()
