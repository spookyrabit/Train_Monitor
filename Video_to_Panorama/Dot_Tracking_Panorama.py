#!/usr/bin/env python3
"""
Train_Panorama_V8.py
Dynamic train panorama generator using fixed scan line and motion tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

# ======================================================
# USER-TUNABLE PARAMETERS
# ======================================================
VIDEO_PATH = "train.mp4"
OUTPUT_PANORAMA = "TEMP/train_panorama.jpg"
OUTPUT_META_CSV = "TEMP/panorama_strips.csv"

FLOW_DOWNSCALE = 0.35
BASE_STRIP_WIDTH = 37
STRIP_MIN = 2
STRIP_MAX = 75
ADAPTIVE_K = 3.0
TARGET_SPACING_PX = 1.0  # horizontal movement between frames in panorama

MAX_FEATURES = 800
FEATURE_QUALITY = 0.01
FEATURE_MIN_DIST = 7
KLT_WIN = (21, 21)
KLT_MAX_LEVEL = 1
KLT_TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

EDGE_CANNY_LOW = 10
EDGE_CANNY_HIGH = 200

SHOW_PREVIEW = True
PREVIEW_FPS = 60
DRAW_FLOW_VECTORS = True

USE_MOTION_MASK = True
MOTION_MASK_HISTORY = 200
MOTION_MASK_VAR_THRESH = 25

REFRESH_FEAT_EVERY = 1
MIN_GOOD_FEATURES = 10
MIN_TRAIN_DX = 2.0  # min px/frame to be considered train
SCAN_X = 0.5  # Fractional x position of fixed scan line (0=left,1=right)
# ======================================================

def detect_features_masked(gray, mask=None):
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=MAX_FEATURES,
        qualityLevel=FEATURE_QUALITY,
        minDistance=FEATURE_MIN_DIST,
        blockSize=7, mask=mask
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2).astype(np.float32)

def compute_fast_features(prev_pts, new_pts, status, min_dx):
    dxs = (new_pts[:,0]-prev_pts[:,0])*status
    mask = np.abs(dxs)>=min_dx
    if not np.any(mask):
        return np.empty((0,2),dtype=np.float32)
    return new_pts[mask]

def edge_density_fullres(full_frame, cx_full, roi_half_w=200):
    h,w = full_frame.shape[:2]
    left = max(0,int(cx_full-roi_half_w))
    right = min(w,int(cx_full+roi_half_w))
    roi = full_frame[:,left:right]
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,EDGE_CANNY_LOW,EDGE_CANNY_HIGH)
    if edges.size==0:
        return 0.0
    return float(edges.sum())/(255.0*edges.size)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video opened: {VIDEO_PATH} | frames={total_frames}")

    ret, full_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    full_h, full_w = full_frame.shape[:2]
    scan_x_px = int(full_w*SCAN_X)

    flow_w = max(32,int(full_w*FLOW_DOWNSCALE))
    flow_h = max(32,int(full_h*FLOW_DOWNSCALE))
    scale_x = full_w/flow_w
    scale_y = full_h/flow_h

    flow_frame = cv2.resize(full_frame,(flow_w,flow_h))
    prev_gray = cv2.cvtColor(flow_frame,cv2.COLOR_BGR2GRAY)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=MOTION_MASK_HISTORY,
        varThreshold=MOTION_MASK_VAR_THRESH,
        detectShadows=False
    ) if USE_MOTION_MASK else None

    mask=None
    if fgbg is not None:
        motion_mask = fgbg.apply(prev_gray)
        _,mask = cv2.threshold(motion_mask,128,255,cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask,5)
    pts_prev = detect_features_masked(prev_gray,mask)

    panorama = []
    meta_rows = []

    cumulative_dx_full=0.0
    frame_idx=1
    refresh_counter=0
    pbar = tqdm(total=total_frames,disable=SHOW_PREVIEW,desc="Panorama")

    cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    while True:
        ret, full_frame = cap.read()
        if not ret:
            break
        frame_idx+=1

        flow_frame = cv2.resize(full_frame,(flow_w,flow_h))
        gray = cv2.cvtColor(flow_frame,cv2.COLOR_BGR2GRAY)

        if pts_prev.size==0:
            mask=None
            if fgbg is not None:
                motion_mask = fgbg.apply(gray)
                _,mask = cv2.threshold(motion_mask,128,255,cv2.THRESH_BINARY)
                mask = cv2.medianBlur(mask,5)
            pts_prev = detect_features_masked(gray,mask)
            if pts_prev.size==0:
                prev_gray=gray.copy()
                if not SHOW_PREVIEW:
                    pbar.update(1)
                continue

        pts_prev_reshaped = pts_prev.reshape(-1,1,2)
        pts_new,status,_ = cv2.calcOpticalFlowPyrLK(
            prev_gray,gray,pts_prev_reshaped,None,
            winSize=KLT_WIN,maxLevel=KLT_MAX_LEVEL,criteria=KLT_TERM
        )
        if pts_new is None:
            pts_prev = detect_features_masked(gray,mask)
            prev_gray = gray.copy()
            if not SHOW_PREVIEW:
                pbar.update(1)
            continue
        pts_new = pts_new.reshape(-1,2)
        status = status.reshape(-1)

        fast_pts = compute_fast_features(pts_prev,pts_new,status,MIN_TRAIN_DX)
        if fast_pts.size==0:
            prev_gray = gray.copy()
            pts_prev = detect_features_masked(gray,mask)
            if not SHOW_PREVIEW:
                pbar.update(1)
            continue

        # Compute strip width dynamically from spread of fast points near scan_x_px
        x_min = np.min(fast_pts[:,0]*scale_x)
        x_max = np.max(fast_pts[:,0]*scale_x)
        strip_width = int(round(BASE_STRIP_WIDTH + ADAPTIVE_K*(x_max-x_min)/flow_w))
        strip_width = np.clip(strip_width,STRIP_MIN,STRIP_MAX)

        left = max(0,scan_x_px-strip_width//2)
        right = min(full_w,left+strip_width)
        if right>left:
            strip = full_frame[:,left:right,:].copy()
            panorama.append(strip)
            cumulative_dx_full += TARGET_SPACING_PX
            meta_rows.append({
                "frame": frame_idx,
                "cx_full": scan_x_px,
                "strip_width": strip_width,
                "good_features": int(fast_pts.shape[0]),
            })

        if SHOW_PREVIEW:
            vis = full_frame.copy()
            cv2.line(vis,(scan_x_px,0),(scan_x_px,full_h),(0,255,0),1)
            if DRAW_FLOW_VECTORS:
                for i,st in enumerate(status):
                    if st==1:
                        x0,y0 = pts_prev[i]
                        x1,y1 = pts_new[i]
                        x0f,y0f = int(x0*scale_x), int(y0*scale_y)
                        x1f,y1f = int(x1*scale_x), int(y1*scale_y)
                        color=(0,255,255) if abs(x1-x0)>1 else (255,0,0)
                        cv2.line(vis,(x0f,y0f),(x1f,y1f),color,1)
                        cv2.circle(vis,(x1f,y1f),2,(0,0,255),-1)
            delay = max(1,int(1000/PREVIEW_FPS))
            cv2.imshow("Preview (ESC to quit)",vis)
            if cv2.waitKey(delay) & 0xFF==27:
                break

        prev_gray = gray.copy()
        pts_prev = pts_new[status==1].astype(np.float32)
        refresh_counter +=1
        if refresh_counter>=REFRESH_FEAT_EVERY or len(pts_prev)<MIN_GOOD_FEATURES:
            mask=None
            if fgbg is not None:
                motion_mask = fgbg.apply(gray)
                _,mask = cv2.threshold(motion_mask,128,255,cv2.THRESH_BINARY)
                mask = cv2.medianBlur(mask,5)
            pts_prev = detect_features_masked(gray,mask)
            refresh_counter=0

        if not SHOW_PREVIEW:
            pbar.update(1)

    if not SHOW_PREVIEW:
        pbar.close()
    else:
        cv2.destroyAllWindows()
    cap.release()

    if len(panorama)>0:
        panorama_img = np.concatenate(panorama,axis=1)
        cv2.imwrite(OUTPUT_PANORAMA,panorama_img,[int(cv2.IMWRITE_JPEG_QUALITY),95])
        print(f"✅ Saved panorama: {Path(OUTPUT_PANORAMA).absolute()} (width={panorama_img.shape[1]} px)")
    else:
        print("⚠️ No strips captured.")

    with open(OUTPUT_META_CSV,"w",newline="") as csvfile:
        fieldnames = ["frame","cx_full","strip_width","good_features"]
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for row in meta_rows:
            writer.writerow(row)
    print(f"✅ Saved metadata CSV: {Path(OUTPUT_META_CSV).absolute()}")
    print(f"Strips captured: {len(panorama)} | Frames processed: {frame_idx}")

if __name__=="__main__":
    main()
