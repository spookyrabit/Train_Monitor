import cv2
import numpy as np
import os
from tqdm import tqdm

# ======================
# SETTINGS
# ======================
INPUT_FOLDER = "TEMP/train_cars"            # folder containing your cropped train car panoramas
OUTPUT_FOLDER = "TEMP/train_cars_cropped"   # folder to save height-cropped images
TOP_PADDING = 20                            # pixels to add above detected top
BOTTOM_PADDING = 10                         # pixels to add below detected rail
RAIL_THRESHOLD_RATIO = 0.99                 # fraction of max Sobel response to detect rail
TOP_DIFF_THRESHOLD = 85                     # threshold for detecting top of car

# ======================
# SETUP
# ======================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def detect_top_bottom(img):
    """Detect approximate top and bottom crop boundaries of a train car image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- TOP DETECTION ---
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
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.abs(sobel_y).sum(axis=1)
    edge_thresh = RAIL_THRESHOLD_RATIO * np.max(edge_strength)

    bottom = h - 1
    for y in range(h - 1, -1, -1):
        if edge_strength[y] > edge_thresh:
            bottom = min(h, y + BOTTOM_PADDING)
            break

    return top, bottom

# ======================
# PROCESS ALL IMAGES
# ======================
invalid_count = 0
total_count = 0

for fname in tqdm(os.listdir(INPUT_FOLDER)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Skipping {fname}: could not read image file.")
        continue

    total_count += 1
    h = img.shape[0]
    top, bottom = detect_top_bottom(img)

    # Clamp boundaries
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))

    cropped = img[top:bottom, :]

    # --- Safety Check ---
    if cropped.size == 0 or cropped.shape[0] <= 1 or cropped.shape[1] <= 1:
        print(f"⚠️ {fname}: invalid crop (top={top}, bottom={bottom}, h={h}) — saving original image instead.")
        cropped = img.copy()
        invalid_count += 1

    out_path = os.path.join(OUTPUT_FOLDER, fname)
    success = cv2.imwrite(out_path, cropped)

    if not success:
        print(f"❌ Failed to save image: {fname}")
    else:
        print(f"✅ {fname}: saved (crop height={cropped.shape[0]}px, original height={h}px)")

# ======================
# DEBUG VISUALIZATION
# ======================
DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug_visual")
os.makedirs(DEBUG_FOLDER, exist_ok=True)

for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    path = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    h = img.shape[0]
    top, bottom = detect_top_bottom(img)
    top = max(0, min(top, h - 1))
    bottom = max(top + 1, min(bottom, h))

    # Determine if fallback was used
    fallback = False
    if bottom - top < 2:
        fallback = True
        top, bottom = 0, h

    vis = img.copy()
    cv2.line(vis, (0, top), (vis.shape[1]-1, top), (0, 255, 0), 2)
    cv2.line(vis, (0, bottom), (vis.shape[1]-1, bottom), (0, 0, 255), 2)
    if fallback:
        cv2.putText(vis, "FALLBACK USED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(DEBUG_FOLDER, fname), vis)

# ======================
# SUMMARY
# ======================
print("\nHeight cropping complete.")
print(f"Cropped images saved in: {OUTPUT_FOLDER}")
print(f"Debug visualizations saved in: {DEBUG_FOLDER}")
print(f"Total images processed: {total_count}")
print(f"Fallback (full image used): {invalid_count}")
print(f"Effective success rate: {100 * (1 - invalid_count / max(1, total_count)):.1f}%")
