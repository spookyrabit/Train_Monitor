import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import csv
import os

# === CONFIG ===
INPUT_PATH = "TEMP/train_panorama.jpg"
OUTPUT_VIS = "TEMP/train_car_boundaries.jpg"
OUTPUT_DIR = "TEMP/train_cars"
OUTPUT_CSV = "TEMP/split_metadata.csv"

GAUSS_BLUR = 5
SAVGOL_WINDOW = 101
SAVGOL_POLY = 3
MIN_PEAK_DISTANCE = 150
MIN_PEAK_PROMINENCE = 10
EDGE_THRESHOLD_FRACTION = 0.55   # relative threshold factor
MIN_SEGMENT_WIDTH_RATIO = 0.5
SUSTAINED_COLS = 10
PADDING = 140

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD IMAGE ===
img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot open {INPUT_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (GAUSS_BLUR, GAUSS_BLUR), 0)

# === EDGE PROFILE ===
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
column_profile = np.abs(sobelx).mean(axis=0)
smoothed = savgol_filter(column_profile, SAVGOL_WINDOW, SAVGOL_POLY)

# === DYNAMIC THRESHOLDING ===
base_level = np.percentile(smoothed, 20)
peak_level = np.percentile(smoothed, 95)
EDGE_THRESHOLD = base_level + EDGE_THRESHOLD_FRACTION * (peak_level - base_level)

# adaptive sanity rescaling
if EDGE_THRESHOLD > peak_level:
    EDGE_THRESHOLD = (base_level + peak_level) / 2
if EDGE_THRESHOLD < base_level + 0.1 * (peak_level - base_level):
    EDGE_THRESHOLD = base_level + 0.15 * (peak_level - base_level)

print(f"[INFO] Adjusted EDGE_THRESHOLD = {EDGE_THRESHOLD:.2f} (base={base_level:.1f}, peak={peak_level:.1f})")

# === DETECT TRAIN START/END ===
train_start = 0
for i in range(len(smoothed)-SUSTAINED_COLS):
    if all(smoothed[i:i+SUSTAINED_COLS] < EDGE_THRESHOLD):
        train_start = i
        break

train_end = len(smoothed)
for i in range(len(smoothed)-SUSTAINED_COLS, 0, -1):
    if all(smoothed[i-SUSTAINED_COLS:i] < EDGE_THRESHOLD):
        train_end = i
        break

# === ADD PADDING ===
train_start = max(0, train_start - PADDING)
train_end = min(len(smoothed), train_end + PADDING)
print(f"[INFO] Train range with padding: {train_start} → {train_end}")

train_profile = smoothed[train_start:train_end]
train_img = img[:, train_start:train_end]

# === DETECT SPLIT PEAKS ===
prominence_val = (peak_level - base_level) * 0.25
peaks, _ = find_peaks(train_profile, distance=MIN_PEAK_DISTANCE, prominence=prominence_val)
peaks = [p for p in peaks if train_profile[p] >= EDGE_THRESHOLD]

# Retry logic
if not peaks:
    print("[WARN] No peaks found with current threshold — retrying with relaxed settings...")
    relax_prom = prominence_val * 0.5
    relax_thresh = base_level + 0.2 * (peak_level - base_level)
    peaks, _ = find_peaks(train_profile, distance=MIN_PEAK_DISTANCE, prominence=relax_prom)
    peaks = [p for p in peaks if train_profile[p] >= relax_thresh]
    print(f"[INFO] Retry found {len(peaks)} peaks (relaxed threshold={relax_thresh:.1f})")
    if not peaks:
        raise RuntimeError("No peaks found after relaxed retry — image may be flat or contrastless.")

boundaries = [0] + sorted(peaks) + [train_end - train_start]
widths = np.diff(boundaries)

# === MEDIAN WIDTH AND OUTLIER FILTER ===
if len(widths) > 2:
    median_w = np.median(sorted(widths)[1:-1])
else:
    median_w = np.median(widths) if len(widths) else 0

filtered_boundaries = [boundaries[0]]
for i in range(1, len(boundaries)):
    w = boundaries[i] - filtered_boundaries[-1]
    if w >= median_w * MIN_SEGMENT_WIDTH_RATIO:
        filtered_boundaries.append(boundaries[i])

boundaries = filtered_boundaries
print(f"[INFO] {len(boundaries)-1} car segments detected after cleanup.")
print(f"[INFO] Estimated median car width ≈ {median_w:.1f}px")

# === MERGE TOO-SMALL SEGMENTS IF CARS ARE UNIFORM ===
widths = np.diff(boundaries)
if len(widths) > 5:
    std_ratio = np.std(widths) / np.mean(widths)
    if std_ratio < 0.2:
        # if cars are fairly uniform, remove spurious internal splits
        new_bounds = [boundaries[0]]
        for i in range(1, len(boundaries)):
            if (boundaries[i] - new_bounds[-1]) >= 0.6 * median_w:
                new_bounds.append(boundaries[i])
        boundaries = new_bounds
        print(f"[INFO] Uniform-length heuristic applied — reduced to {len(boundaries)-1} cars.")

# === DRAW VISUALIZATION ===
vis = train_img.copy()
for i, x in enumerate(boundaries):
    color = (0, 255, 0) if i in (0, len(boundaries)-1) else (0, 0, 255)
    cv2.line(vis, (int(x), 0), (int(x), vis.shape[0]), color, 2)
cv2.imwrite(OUTPUT_VIS, vis)
print(f"Saved visualization → {OUTPUT_VIS}")

# === CROP AND SAVE EACH CAR ===
csv_data = [["car_id", "x1", "x2", "width_px", "filename"]]
car_num = 1

for i in range(len(boundaries)-1):
    x1, x2 = int(boundaries[i]), int(boundaries[i+1])
    w = x2 - x1
    if w < median_w * MIN_SEGMENT_WIDTH_RATIO:
        continue
    car_img = train_img[:, x1:x2]
    out_file = f"car_{car_num:02d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_file)
    cv2.imwrite(out_path, car_img)
    csv_data.append([car_num, x1 + train_start, x2 + train_start, w, out_file])
    print(f"Saved: {out_file} ({w}px wide)")
    car_num += 1

# === SAVE CSV METADATA ===
with open(OUTPUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows(csv_data)
print(f"Metadata saved → {OUTPUT_CSV}")

# === DEBUG PLOT ===
plt.figure(figsize=(14, 6))
plt.plot(smoothed, label="Smoothed profile", color="orange")
plt.scatter([p + train_start for p in peaks],
            [smoothed[p + train_start] for p in peaks],
            c='red', s=40, label="Peaks")
plt.axvline(train_start, color='green', linestyle='--', label="Train start (padded)")
plt.axvline(train_end, color='green', linestyle='--', label="Train end (padded)")
plt.title("v12e: Train Car Boundaries (Adaptive Threshold + Uniform Merge)")
plt.legend()
plt.tight_layout()
plt.savefig("TEMP/edge_profile.png", dpi=200)
plt.close()
