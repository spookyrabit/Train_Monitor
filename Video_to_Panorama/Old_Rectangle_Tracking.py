#!/usr/bin/env python3
"""
Train_Panorama_ShapeTrack_v2_rect_gpu.py

Rectangle-only shape tracker + panorama builder with optional CUDA edge detector
and debug video output.

Outputs:
 - Full-resolution panorama JPG
 - Metadata CSV
 - Debug overlay video (same FPS & resolution as source)

Usage:
  python Train_Panorama_ShapeTrack_v2_rect_gpu.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import math
import time

# -------------------------
# USER-TUNABLE PARAMETERS
# -------------------------
VIDEO_PATH = "train.mp4"

OUTPUT_PANORAMA = "TEMP/train_panorama.jpg"
OUTPUT_META = "TEMP/panorama_meta.csv"
OUTPUT_DEBUG_VIDEO = "TEMP/debug_shapetrack.mp4"

# Detection/resolution
DETECT_DOWNSCALE = 1.0    # detection scale (down from full-res)
MIN_RECT_AREA = 250        # min contour area (on downscaled)
RECT_APPROX_EPS = 0.05     # approxPolyDP epsilon factor
MIN_ASPECT = 0.05           # allowed rectangle aspect ratio (w/h) min
MAX_ASPECT = 20.0           # max aspect ratio
MIN_SOLIDITY = 0.1         # contour solidity (area / hull area)

# Matching and motion constraints (in downscale-space)
MAX_MATCH_DIST = 80.0
MIN_RECT_DX = 5.0          # minimal dx to consider movement
MAX_POINT_DX = 120.0       # reject teleporting matches
MAX_ANGLE_DEG = 30.0       # mostly horizontal movement

# Panorama sampling
SCAN_X = 0.5               # fraction across full width where we sample
TARGET_SPACING_PX = 2.0    # spacing in full-res pixels between appended columns
BASE_STRIP_WIDTH = 20
STRIP_MIN = 2
STRIP_MAX = 60
ADAPTIVE_K = 0.35

# Preview & debug
SHOW_PREVIEW = True       # still produce debug video even if preview off
PREVIEW_FPS = 30            # displayed preview frame rate (affects waitKey throttle)
DRAW_MATCHES = True
DRAW_RECTS = True

# GPU / CUDA
FORCE_CPU = False          # set True to avoid CUDA even if available

# Smoothing
EMA_ALPHA = 0.25           # exponential moving average alpha for med_dx smoothing

# Other control
REFRESH_EVERY = 5
# -------------------------

def has_cuda():
    try:
        if FORCE_CPU:
            return False
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

USE_CUDA = has_cuda()
print(f"CUDA available: {USE_CUDA}")

# Create CUDA Canny detector object if possible
cuda_canny = None
if USE_CUDA:
    try:
        # createCannyEdgeDetector exists in many OpenCV CUDA builds
        cuda_canny = cv2.cuda.createCannyEdgeDetector(50, 150)
    except Exception:
        cuda_canny = None
        USE_CUDA = False

# -------------------------
# Helper functions
# -------------------------
def preprocess_downscale(full, det_w, det_h):
    det = cv2.resize(full, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(det, cv2.COLOR_BGR2GRAY)
    # contrast + small bilateral to keep edges
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    return det, gray

def edges_cuda(gray):
    """Run Canny on GPU; expects 8-bit gray ndarray."""
    gpu = cv2.cuda_GpuMat()
    gpu.upload(gray)
    # convert to UMat or appropriate? cuda Canny returns gpu mat
    edges_gpu = cuda_canny.detect(gpu)
    return edges_gpu.download()

def edges_cpu(gray):
    return cv2.Canny(gray, 80, 200)

def find_rectangles(edges, det_img):
    """Find rectangles (approx poly with 4 verts, solidity and aspect checks)."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    h_det, w_det = det_img.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_RECT_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, RECT_APPROX_EPS * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # bounding and metrics
        xs = approx[:,0,0]; ys = approx[:,0,1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        w = float(x2 - x1); h = float(y2 - y1)
        if h <= 0 or w <= 0:
            continue
        aspect = w / h
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) if hull is not None else 0
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue
        cx = float(xs.mean()); cy = float(ys.mean())
        rects.append({
            "type":"rect",
            "centroid": (cx, cy),
            "w": w, "h": h, "area": area,
            "bbox": (x1,y1,x2,y2),
            "poly": approx
        })
    return rects

def match_rects(prev, cur, max_dist=MAX_MATCH_DIST):
    """Greedy nearest-neighbor matching by centroid distance; returns matches and unmatched."""
    if len(prev) == 0 or len(cur) == 0:
        return [], list(range(len(prev))), list(range(len(cur)))
    prev_cent = np.array([p["centroid"] for p in prev])
    cur_cent = np.array([c["centroid"] for c in cur])
    dists = np.linalg.norm(prev_cent[:,None,:] - cur_cent[None,:,:], axis=2)
    matches = []
    used_prev = set(); used_cur = set()
    INF = np.max(dists) + 1.0
    while True:
        # get minimal distance
        idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        p,c = idx
        if dists[p,c] == INF:
            break
        if dists[p,c] > max_dist:
            break
        matches.append((p,c))
        used_prev.add(p); used_cur.add(c)
        dists[p,:] = INF
        dists[:,c] = INF
        if np.all(dists == INF):
            break
    unmatched_prev = [i for i in range(len(prev)) if i not in used_prev]
    unmatched_cur = [j for j in range(len(cur)) if j not in used_cur]
    return matches, unmatched_prev, unmatched_cur

# -------------------------
# Main
# -------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Opened {VIDEO_PATH} | frames={total_frames} | fps={src_fps:.2f}")

    # read first frame
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    full_h, full_w = first.shape[:2]
    det_w = max(64, int(full_w * DETECT_DOWNSCALE))
    det_h = max(64, int(full_h * DETECT_DOWNSCALE))
    scale_x = float(full_w) / det_w
    scale_y = float(full_h) / det_h
    scan_x_px = int(full_w * SCAN_X)

    print(f"Full-res {full_w}x{full_h}, detect {det_w}x{det_h}, scale_x={scale_x:.3f}")

    # Debug video writer (same size and fps as source)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    debug_writer = cv2.VideoWriter(OUTPUT_DEBUG_VIDEO, fourcc, src_fps, (full_w, full_h))

    # state
    prev_rects = []
    panorama_strips = []
    meta = []
    ema_dx = 0.0
    first_measure = True
    cumulative_distance_full = 0.0

    # rewind
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=total_frames, desc="ShapePanoramaV2")

    frame_idx = 0
    last_time = time.time()
    while True:
        ret, full = cap.read()
        if not ret:
            break
        frame_idx += 1

        # detection downscale + preprocess
        det, gray_det = preprocess_downscale(full, det_w, det_h)
        if USE_CUDA and cuda_canny is not None:
            edges = edges_cuda(gray_det)
        else:
            edges = edges_cpu(gray_det)

        # rectify small morphological holes on CPU (helps contour extraction)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        rects = find_rectangles(edges, det)

        # match prev rects -> cur rects
        matches, unmatched_prev, unmatched_cur = match_rects(prev_rects, rects, max_dist=MAX_MATCH_DIST)

        # compute dxs for matches and filter by angle & magnitude
        dxs = []
        valid_pairs = []
        for (p_idx, c_idx) in matches:
            p = prev_rects[p_idx]["centroid"]
            c = rects[c_idx]["centroid"]
            dx = c[0] - p[0]
            dy = c[1] - p[1]
            angle = abs(math.degrees(math.atan2(dy, dx))) if dx != 0 else 90.0
            if abs(dx) <= MAX_POINT_DX and abs(angle) <= MAX_ANGLE_DEG:
                dxs.append(dx)
                valid_pairs.append((p_idx, c_idx, dx))

        if len(dxs) > 0:
            med_dx_det = float(np.median(dxs))
            med_dx_full = med_dx_det * scale_x
            # EMA smoothing
            if first_measure:
                ema_dx = med_dx_full
                first_measure = False
            else:
                ema_dx = EMA_ALPHA * med_dx_full + (1.0 - EMA_ALPHA) * ema_dx
        else:
            med_dx_det = 0.0
            med_dx_full = 0.0
            # slowly decay ema towards zero
            ema_dx = (1.0 - EMA_ALPHA) * ema_dx

        # accumulate and sample strips based on ema_dx
        cumulative_distance_full += abs(ema_dx)
        # compute adaptive strip width from rectangles near scan line
        scan_half_window_full = 400
        near_rects = []
        for r in rects:
            cx_det, cy_det = r["centroid"]
            cx_full = cx_det * scale_x
            if abs(cx_full - scan_x_px) <= scan_half_window_full:
                near_rects.append(r)

        if len(near_rects) > 0:
            xs_full = [r["centroid"][0] * scale_x for r in near_rects]
            spread = max(xs_full) - min(xs_full) if len(xs_full) > 1 else 0.0
            strip_w = int(round(BASE_STRIP_WIDTH + ADAPTIVE_K * spread))
        else:
            strip_w = BASE_STRIP_WIDTH
        strip_w = int(np.clip(strip_w, STRIP_MIN, STRIP_MAX))

        if cumulative_distance_full >= TARGET_SPACING_PX:
            left = int(np.clip(scan_x_px - strip_w // 2, 0, full_w - 1))
            right = int(np.clip(left + strip_w, 1, full_w))
            if right > left:
                strip = full[:, left:right, :].copy()
                panorama_strips.append(strip)
                meta.append({
                    "frame": frame_idx,
                    "scan_x": scan_x_px,
                    "strip_w": strip_w,
                    "ema_dx": float(round(ema_dx,3)),
                    "near_rects": len(near_rects)
                })
            cumulative_distance_full = 0.0

        # produce debug frame overlay
        debug = full.copy()
        # draw scan line
        cv2.line(debug, (scan_x_px, 0), (scan_x_px, full_h), (0, 200, 0), 1)
        # draw rectangles (scaled to full-res)
        for i, r in enumerate(rects):
            x1,y1,x2,y2 = r["bbox"]
            x1f = int(round(x1 * scale_x)); y1f = int(round(y1 * scale_y))
            x2f = int(round(x2 * scale_x)); y2f = int(round(y2 * scale_y))
            color = (0, 255, 255) if abs((r["centroid"][0] * scale_x) - scan_x_px) < scan_half_window_full else (200, 200, 200)
            cv2.rectangle(debug, (x1f, y1f), (x2f, y2f), color, 2)
            cx_f = int(round(r["centroid"][0] * scale_x)); cy_f = int(round(r["centroid"][1] * scale_y))
            cv2.circle(debug, (cx_f, cy_f), 3, (255, 255, 0), -1)
        # draw matches
        if DRAW_MATCHES:
            for (p_idx, c_idx) in matches:
                p = prev_rects[p_idx]["centroid"]
                c = rects[c_idx]["centroid"]
                p_f = (int(round(p[0] * scale_x)), int(round(p[1] * scale_y)))
                c_f = (int(round(c[0] * scale_x)), int(round(c[1] * scale_y)))
                # color valid pairs green, invalid red
                valid = any((p_idx == vp and c_idx == vc) for vp,vc,_ in valid_pairs)
                col = (0, 255, 0) if valid else (0, 0, 255)
                cv2.line(debug, p_f, c_f, col, 1)
                cv2.circle(debug, c_f, 3, col, -1)
        # overlay info
        cv2.putText(debug, f"frame {frame_idx}/{total_frames} | ema_dx={ema_dx:.2f}px | rects={len(rects)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # write debug video
        debug_writer.write(debug)

        # show preview if requested
        if SHOW_PREVIEW:
            cv2.imshow("ShapeTrackV2 Debug", debug)
            if cv2.waitKey(int(max(1, 1000.0 / PREVIEW_FPS))) & 0xFF == 27:
                break

        # prepare for next frame
        prev_rects = rects
        if frame_idx % REFRESH_EVERY == 0:
            # small no-op: this forces fresh detection next frame (we re-detect every frame anyway)
            pass

        pbar.update(1)

    # cleanup
    pbar.close()
    cap.release()
    debug_writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # save panorama
    if len(panorama_strips) > 0:
        pano = np.concatenate(panorama_strips, axis=1)
        cv2.imwrite(OUTPUT_PANORAMA, pano, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"Saved panorama: {Path(OUTPUT_PANORAMA).absolute()} width={pano.shape[1]}")
    else:
        print("No strips captured; adjust thresholds or spacing")

    # save metadata CSV
    with open(OUTPUT_META, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","scan_x","strip_w","ema_dx","near_rects"])
        writer.writeheader()
        for r in meta:
            writer.writerow(r)
    print(f"Saved metadata CSV: {Path(OUTPUT_META).absolute()}")

if __name__ == "__main__":
    main()
