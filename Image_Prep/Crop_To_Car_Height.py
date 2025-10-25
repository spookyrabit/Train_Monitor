import cv2
import numpy as np
import os
from tqdm import tqdm

# ======================
# SETTINGS
# ======================
INPUT_FOLDER = "TEMP/train_cars"      # folder containing your cropped train car panoramas
OUTPUT_FOLDER = "TEMP/train_cars_cropped"  # folder to save height-cropped images
TOP_PADDING = 20                        # pixels to add above detected top
BOTTOM_PADDING = 10                     # pixels to add below detected rail
RAIL_THRESHOLD_RATIO = 0.99             # fraction of max Sobel response to detect rail
TOP_DIFF_THRESHOLD = 85                # threshold for detecting top of car

# ======================
# SETUP
# ======================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_top_bottom(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- TOP DETECTION ---
    # Use top few rows as background reference
    bg_rows = gray[0:10, :]
    bg_mean = np.mean(bg_rows, axis=0)

    top = 0
    for y in range(h):
        row = gray[y, :]
        diff = np.mean(np.abs(row - bg_mean))
        if diff > TOP_DIFF_THRESHOLD:
            top = max(0, y - TOP_PADDING)
            break

    # --- BOTTOM DETECTION (RAIL) ---
    # Sobel vertical edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.abs(sobel_y).sum(axis=1)
    edge_thresh = RAIL_THRESHOLD_RATIO * np.max(edge_strength)

    bottom = h - 1
    for y in range(h-1, -1, -1):
        if edge_strength[y] > edge_thresh:
            bottom = min(h, y + BOTTOM_PADDING)
            break

    return top, bottom

# ======================
# PROCESS ALL IMAGES
# ======================
for fname in tqdm(os.listdir(INPUT_FOLDER)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    top, bottom = detect_top_bottom(img)
    cropped = img[top:bottom, :]
    out_path = os.path.join(OUTPUT_FOLDER, fname)
    cv2.imwrite(out_path, cropped)

    # Debug info
    print(f"{fname}: top={top}, bottom={bottom}, height={bottom-top}px")

# ======================
# OPTIONAL: SAVE DEBUG VISUALIZATION
# ======================
# Draw lines on the original images and save them
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug_visual")
os.makedirs(DEBUG_FOLDER, exist_ok=True)

for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        continue
    top, bottom = detect_top_bottom(img)
    vis = img.copy()
    cv2.line(vis, (0, top), (vis.shape[1]-1, top), (0, 255, 0), 2)
    cv2.line(vis, (0, bottom), (vis.shape[1]-1, bottom), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(DEBUG_FOLDER, fname), vis)

print("Height cropping complete. Cropped images saved in:", OUTPUT_FOLDER)
print("Debug visualizations saved in:", DEBUG_FOLDER)
