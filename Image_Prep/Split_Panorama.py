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
OUTPUT_CSV = "TEMP/train_cars_metadata.csv"

GAUSS_BLUR = 5
SAVGOL_WINDOW = 101
SAVGOL_POLY = 3
MIN_PEAK_DISTANCE = 150
MIN_PEAK_PROMINENCE = 10
EDGE_THRESHOLD = 27          # Threshold to detect train edges
MIN_SEGMENT_WIDTH_RATIO = 0.5  # Minimum fraction of median width to keep a segment
SUSTAINED_COLS = 10          # Number of consecutive columns below threshold to define train start/end
PADDING = 140                 # pixels to add back at start/end

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

# === DETECT TRAIN START (FRONT) ===
train_start = 0
for i in range(len(smoothed)-SUSTAINED_COLS):
    if all(smoothed[i:i+SUSTAINED_COLS] < EDGE_THRESHOLD):
        train_start = i
        break

# === DETECT TRAIN END (BACK) ===
train_end = len(smoothed)
for i in range(len(smoothed)-SUSTAINED_COLS, 0, -1):
    if all(smoothed[i-SUSTAINED_COLS:i] < EDGE_THRESHOLD):
        train_end = i
        break

# === ADD PADDING ===
train_start = max(0, train_start - PADDING)
train_end = min(len(smoothed), train_end + PADDING)

print(f"[INFO] Train range with padding: {train_start} → {train_end}")

# Crop to train body
train_profile = smoothed[train_start:train_end]
train_img = img[:, train_start:train_end]

# === FIND CAR SPLIT PEAKS ===
peaks, _ = find_peaks(
    train_profile,
    distance=MIN_PEAK_DISTANCE,
    prominence=MIN_PEAK_PROMINENCE
)
peaks = [p for p in peaks if train_profile[p] >= EDGE_THRESHOLD]
boundaries = [0] + sorted(peaks) + [train_end - train_start]

# Compute median width
widths = np.diff(boundaries)
if len(widths) > 2:
    median_w = np.median(sorted(widths)[1:-1])
else:
    median_w = np.median(widths) if len(widths) else 0

# Remove too-small segments (inside train body)
filtered_boundaries = [boundaries[0]]
for i in range(1, len(boundaries)):
    if boundaries[i] - filtered_boundaries[-1] >= median_w * MIN_SEGMENT_WIDTH_RATIO:
        filtered_boundaries.append(boundaries[i])
boundaries = filtered_boundaries

print(f"[INFO] {len(boundaries)-1} car segments detected after cleanup.")

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
        continue  # Skip tiny segment
    car_img = train_img[:, x1:x2]
    out_file = f"car_{car_num:02d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_file)
    cv2.imwrite(out_path, car_img)
    csv_data.append([car_num, x1 + train_start, x2 + train_start, w, out_file])
    print(f"Saved: {out_file} ({w}px wide)")
    car_num += 1

# Save CSV metadata
with open(OUTPUT_CSV, "w", newline="") as f:
    csv.writer(f).writerows(csv_data)
print(f"Metadata saved → {OUTPUT_CSV}")

# === DEBUG PLOT ===
plt.figure(figsize=(14, 6))
plt.plot(smoothed, label="Smoothed profile", color="orange")
plt.scatter([p + train_start for p in peaks], [smoothed[p + train_start] for p in peaks], c='red', s=40, label="Peaks")
plt.axvline(train_start, color='green', linestyle='--', label="Train start (padded)")
plt.axvline(train_end, color='green', linestyle='--', label="Train end (padded)")
plt.title("v12a: Train Car Boundaries with Padding")
plt.legend()
plt.tight_layout()
plt.savefig("TEMP/edge_profile.png", dpi=200)
plt.close()
