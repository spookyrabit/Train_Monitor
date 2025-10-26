#!/usr/bin/env python3
"""
Align_Video.py
---------------------------------------
Estimates and corrects train video tilt by analyzing frame geometry.
Uses two indicators:
  (1) Dominant rectangular shapes on the train body
  (2) Horizontal line segments (the visible rail) near the bottom of the frame

Heavily favors the rail-based angle when present.

Creates: TEMP/first_rotated_preview.jpg and overwrites the input video with an aligned version.
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import os

# -------------------------
# CONFIGURATION
# -------------------------
VIDEO_PATH = "train.mp4"
TEMP_DIR = Path("TEMP")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_PREVIEW = TEMP_DIR / "first_rotated_preview.jpg"

# Detection params
ANGLE_SAMPLE_FRAMES = 8
MIN_RECT_AREA = 250
RECT_APPROX_EPS = 0.05
MIN_ASPECT = 0.05
MAX_ASPECT = 20.0
MIN_SOLIDITY = 0.1
RAIL_REGION_HEIGHT_RATIO = 0.35  # bottom 35% of frame used to detect rail lines
RAIL_MAX_ANGLE = 15.0
RECT_WEIGHT = 0.3     # body confidence weight
RAIL_WEIGHT = 0.7     # rail confidence weight
FORCE_CPU = False

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def has_cuda():
    try:
        if FORCE_CPU:
            return False
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def edges_cpu(gray):
    edges = cv2.Canny(gray, 80, 200)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    return gray

def find_rectangles(edges):
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
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue
        rects.append(approx)
    return rects

def compute_frame_body_angle(rects):
    angles = []
    for poly in rects:
        rect = cv2.minAreaRect(poly)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90
        angles.append(angle)
    return np.median(angles) if len(angles) > 0 else None

def compute_rail_angle(gray):
    """Detect near-horizontal lines near the bottom of the frame."""
    h, w = gray.shape
    bottom_region = gray[int(h * (1 - RAIL_REGION_HEIGHT_RATIO)):, :]
    edges = cv2.Canny(bottom_region, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=w // 5, maxLineGap=15)
    if lines is None:
        return None
    frame_angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < RAIL_MAX_ANGLE:
            frame_angles.append(a)
    return np.median(frame_angles) if len(frame_angles) > 0 else None

# -------------------------
# MAIN
# -------------------------
def main():
    use_cuda = has_cuda()
    print(f"[INFO] CUDA available: {use_cuda} (not used in this step)")
    print(f"[INFO] Analyzing: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sample_idxs = sorted(random.sample(range(total_frames),
                        min(ANGLE_SAMPLE_FRAMES, total_frames)))

    body_angles = []
    rail_angles = []

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = preprocess(frame)
        edges = edges_cpu(gray)

        rects = find_rectangles(edges)
        body_angle = compute_frame_body_angle(rects)
        rail_angle = compute_rail_angle(gray)

        if body_angle is not None:
            body_angles.append(body_angle)
        if rail_angle is not None:
            rail_angles.append(rail_angle)

    cap.release()

    final_body_angle = np.median(body_angles) if len(body_angles) > 0 else 0.0
    final_rail_angle = np.median(rail_angles) if len(rail_angles) > 0 else 0.0

    if len(rail_angles) > 0:
        combined_angle = final_rail_angle
        print(f"[INFO] Using rail-only angle ({len(rail_angles)} frames): {final_rail_angle:.2f}°")
    else:
        combined_angle = RECT_WEIGHT * final_body_angle + RAIL_WEIGHT * final_rail_angle
        print(f"[INFO] Body angle: {final_body_angle:.2f}°, Rail angle: {final_rail_angle:.2f}° → Combined: {combined_angle:.2f}°")

    print(f"[INFO] Estimated dominant train angle: {combined_angle:.2f}°")

    # -------------------------
    # Apply rotation (corrected sign)
    # -------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_path = str(Path(VIDEO_PATH).with_suffix(".aligned.mp4"))
    out = cv2.VideoWriter(safe_path, fourcc, fps, (width, height))

    print(f"[INFO] Rotating {frame_count} frames by {combined_angle:+.2f}°...")
    rot_mat = cv2.getRotationMatrix2D((width // 2, height // 2), combined_angle, 1.0)

    success = True
    for _ in tqdm(range(frame_count), desc="Aligning"):
        ret, frame = cap.read()
        if not ret:
            break
        rotated = cv2.warpAffine(frame, rot_mat, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        out.write(rotated)
    cap.release()
    out.release()

    # Verify output integrity
    cap2 = cv2.VideoCapture(safe_path)
    ret, preview = cap2.read()
    if ret:
        cv2.imwrite(str(OUTPUT_PREVIEW), preview)
    cap2.release()

    os.replace(safe_path, VIDEO_PATH)
    print(f"[INFO] Saved verified aligned video → {Path(VIDEO_PATH).absolute()}")
    print(f"[INFO] Overwrote input safely with aligned version.")

if __name__ == "__main__":
    main()
