import cv2
import json
import os
import csv
import numpy as np

LABELS_FILE = "TEMP/train_car_labels.csv"
IMAGES_FOLDER = "TEMP/train_cars_cropped"
OUTPUT_FILE = "Settings/New_car_region.json"

# --- Load data ---
with open(LABELS_FILE, newline='') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# --- Group by label ---
label_groups = {}
for row in data:
    label = row["label"]
    label_groups.setdefault(label, []).append(row["filename"])

print("\n📸 Define ID regions for each label type.")
print("Instructions:")
print("  → Click and drag with LEFT mouse button to draw the ID area.")
print("  → Press ENTER (RETURN) to accept and move to next image.")
print("  → Press 'r' to reset the selection for the current image.")
print("  → Press ESC to skip this image.\n")

# --- Global vars used in the mouse callback ---
drawing = False
start_point = None
end_point = None
current_img = None
roi = None

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, current_img, roi
    temp_img = current_img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        roi = (start_point, end_point)
        cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_img)

# --- Results dict: label -> list of boxes ---
results = {}

for label, filenames in label_groups.items():
    print(f"🟦 Label: {label} ({len(filenames)} examples)")
    boxes = []

    for fname in filenames:
        img_path = os.path.join(IMAGES_FOLDER, fname)
        if not os.path.exists(img_path):
            print(f"⚠️ Missing {img_path}, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}, skipping.")
            continue

        h, w = img.shape[:2]

        current_img = img.copy()
        roi = None

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select ROI", draw_rectangle)

        while True:
            cv2.imshow("Select ROI", current_img)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER
                if roi is not None:
                    (x1, y1), (x2, y2) = roi
                    # Normalize to [0,1]
                    x1n, y1n, x2n, y2n = x1/w, y1/h, x2/w, y2/h
                    boxes.append((x1n, y1n, x2n, y2n))
                    print(f"✅ Saved ROI for {fname}")
                    break
                else:
                    print("⚠️ No ROI selected, skipping.")
                    break
            elif key == ord('r'):
                roi = None
                current_img = img.copy()
                print("↩️  Reset current selection.")
            elif key == 27:  # ESC
                print("⏭️  Skipped this image.")
                break

        cv2.destroyAllWindows()

    if boxes:
        arr = np.array(boxes)
        avg_box = arr.mean(axis=0)
        results[label] = {
            "x1": float(avg_box[0]),
            "y1": float(avg_box[1]),
            "x2": float(avg_box[2]),
            "y2": float(avg_box[3]),
        }
        print(f"📊 Averaged box for {label}: {results[label]}")
    else:
        print(f"⚠️ No boxes selected for {label}")

# --- Save results ---
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n✅ Saved averaged region definitions → {OUTPUT_FILE}")
