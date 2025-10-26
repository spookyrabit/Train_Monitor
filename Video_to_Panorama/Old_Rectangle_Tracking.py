#!/usr/bin/env python3
"""
Train_Panorama_ShapeTrack_v3_rect_noalign.py

Rectangle-only shape tracker + panorama builder.
This version removes the automatic horizontal alignment / rotation step.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import time
from collections import deque
import random

# -------------------------
# USER-TUNABLE PARAMETERS
# -------------------------
VIDEO_PATH = "train.mp4"

OUTPUT_PANORAMA = "TEMP/train_panorama.jpg"
OUTPUT_META = "TEMP/panorama_meta.csv"
OUTPUT_DEBUG_VIDEO = "TEMP/debug_shapetrack.mp4"

# Detection/resolution
DETECT_DOWNSCALE = 1.0
MIN_RECT_AREA = 250
RECT_APPROX_EPS = 0.05
MIN_ASPECT = 0.05
MAX_ASPECT = 20.0
MIN_SOLIDITY = 0.1

# Matching and motion constraints
MAX_MATCH_DIST = 80.0
MAX_POINT_DX = 120.0
MAX_ANGLE_DEG = 30.0

# Panorama sampling
SCAN_X = 0.5
TARGET_SPACING_PX = 2.0
BASE_STRIP_WIDTH = 20
STRIP_MIN = 2
STRIP_MAX = 60
ADAPTIVE_K = 0.35

# Flow / speed tuning
FLOW_ROI_W = 256
SPEED_SCALE = 10.0
MIN_MOVEMENT_PX = 2.0

# Preview & debug
SHOW_PREVIEW = False
PREVIEW_FPS = 300
DRAW_MATCHES = True
DRAW_RECTS = True

# GPU / CUDA
FORCE_CPU = False

# Smoothing
EMA_ALPHA = 0.25

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

cuda_canny = None
if USE_CUDA:
    try:
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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    return det, gray

def edges_cuda(gray):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(gray)
    edges_gpu = cuda_canny.detect(gpu)
    return edges_gpu.download()

def edges_cpu(gray):
    return cv2.Canny(gray, 80, 200)

def find_rectangles(edges, det_img):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_RECT_AREA:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, RECT_APPROX_EPS * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        xs = approx[:, 0, 0]
        ys = approx[:, 0, 1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        w = float(x2 - x1)
        h = float(y2 - y1)
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
        cx = float(xs.mean())
        cy = float(ys.mean())
        rects.append({
            "type": "rect",
            "centroid": (cx, cy),
            "w": w, "h": h, "area": area,
            "solidity": solidity,
            "bbox": (x1, y1, x2, y2),
            "poly": approx
        })
    return rects

def match_rects(prev, cur, max_dist=MAX_MATCH_DIST):
    if len(prev) == 0 or len(cur) == 0:
        return [], list(range(len(prev))), list(range(len(cur)))
    prev_cent = np.array([p["centroid"] for p in prev])
    cur_cent = np.array([c["centroid"] for c in cur])
    dists = np.linalg.norm(prev_cent[:, None, :] - cur_cent[None, :, :], axis=2)
    matches = []
    used_prev = set()
    used_cur = set()
    INF = np.max(dists) + 1.0
    while True:
        idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        p, c = idx
        if dists[p, c] == INF:
            break
        if dists[p, c] > max_dist:
            break
        matches.append((p, c))
        used_prev.add(p)
        used_cur.add(c)
        dists[p, :] = INF
        dists[:, c] = INF
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

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    full_h, full_w = first.shape[:2]
    det_w = max(64, int(full_w * DETECT_DOWNSCALE))
    det_h = max(64, int(full_h * DETECT_DOWNSCALE))
    scale_x = float(full_w) / det_w
    scale_y = float(full_h) / det_h
    scan_x_px = int(full_w * SCAN_X)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    debug_writer = cv2.VideoWriter(OUTPUT_DEBUG_VIDEO, fourcc, src_fps, (full_w, full_h))

    prev_rects = []
    panorama_strips = deque()
    meta = []
    ema_dx = 0.0
    first_measure = True
    cumulative_distance_full = 0.0
    prev_gray_full = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=total_frames, desc="Shape to Panorama")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        det, gray_det = preprocess_downscale(frame, det_w, det_h)
        edges = edges_cpu(gray_det) if not USE_CUDA else edges_cuda(gray_det)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        rects = find_rectangles(edges, frame)
        matches, _, _ = match_rects(prev_rects, rects)

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow ROI
        roi_half = int(min(FLOW_ROI_W//2, full_w//2-1))
        left_roi = max(0, scan_x_px - roi_half)
        right_roi = min(full_w, scan_x_px + roi_half)
        prev_roi = prev_gray_full[:, left_roi:right_roi]
        curr_roi = gray_full[:, left_roi:right_roi]

        measured_dx = 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None,
                                                pyr_scale=0.5, levels=3, winsize=21,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            measured_dx = float(np.median(flow[...,0]))
        except Exception:
            measured_dx = 0.0

        prev_gray_full = gray_full
        if first_measure:
            ema_dx = measured_dx
            first_measure = False
        else:
            ema_dx = EMA_ALPHA * measured_dx + (1-EMA_ALPHA)*ema_dx

        effective_dx = ema_dx if abs(ema_dx) >= MIN_MOVEMENT_PX else 0.0
        cumulative_distance_full += abs(effective_dx)

        # Adaptive strip width
        speed = abs(effective_dx)
        scale = 1.0 / (0.2 + (speed / SPEED_SCALE))
        strip_w = int(np.clip(round(BASE_STRIP_WIDTH * scale), STRIP_MIN, STRIP_MAX))
        scan_half_window_full = 400
        near_rects = [r for r in rects if abs(r["centroid"][0]*scale_x - scan_x_px) <= scan_half_window_full]
        if len(near_rects) > 0:
            xs_full = [r["centroid"][0]*scale_x for r in near_rects]
            spread = max(xs_full) - min(xs_full) if len(xs_full) > 1 else 0.0
            strip_w = int(np.clip(round(strip_w + ADAPTIVE_K*spread), STRIP_MIN, STRIP_MAX))

        if cumulative_distance_full >= TARGET_SPACING_PX:
            left = int(np.clip(scan_x_px - strip_w//2, 0, full_w-1))
            right = int(np.clip(left + strip_w, 1, full_w))
            if right > left:
                strip = frame[:, left:right, :].copy()
                if effective_dx >= 0:
                    panorama_strips.appendleft(strip)
                else:
                    panorama_strips.append(strip)

                meta.append({
                    "frame": frame_idx,
                    "scan_x": scan_x_px,
                    "strip_w": strip_w,
                    "ema_dx": float(round(ema_dx,4)),
                    "measured_dx": float(round(measured_dx,4)),
                    "near_rects": len(near_rects),
                    "dir": int(np.sign(effective_dx))
                })
            cumulative_distance_full = 0.0

        # Debug overlay
        debug = frame.copy()
        cv2.line(debug, (scan_x_px,0),(scan_x_px,full_h),(0,200,0),1)
        for r in rects:
            x1,y1,x2,y2 = r["bbox"]
            x1f = int(round(x1*scale_x)); y1f=int(round(y1*scale_y))
            x2f = int(round(x2*scale_x)); y2f=int(round(y2*scale_y))
            color = (0,255,255) if abs((r["centroid"][0]*scale_x)-scan_x_px)<scan_half_window_full else (200,200,200)
            cv2.rectangle(debug,(x1f,y1f),(x2f,y2f),color,2)
        debug_writer.write(debug)
        if SHOW_PREVIEW:
            cv2.imshow("ShapeTrackV3 Debug (No Align)", debug)
            if cv2.waitKey(int(max(1,1000/PREVIEW_FPS)))&0xFF==27:
                break

        prev_rects = rects
        pbar.update(1)

    # Cleanup
    pbar.close()
    cap.release()
    debug_writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    if len(panorama_strips)>0:
        pano = np.concatenate(list(panorama_strips), axis=1)
        cv2.imwrite(OUTPUT_PANORAMA, pano,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        print(f"Saved panorama: {Path(OUTPUT_PANORAMA).absolute()} width={pano.shape[1]}")
    else:
        print("No strips captured; adjust thresholds or spacing")

    with open(OUTPUT_META,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame","scan_x","strip_w","ema_dx","measured_dx","near_rects","dir"])
        writer.writeheader()
        for r in meta:
            writer.writerow(r)
    print(f"Saved metadata CSV: {Path(OUTPUT_META).absolute()}")

if __name__=="__main__":
    main()
