import os
import json
import csv
import cv2
import torch
import numpy as np
import easyocr
import re

# ========== GPU AUTO-FALLBACK HANDLER ==========
def init_easyocr():
    use_gpu = False
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                print(f"🟢 GPU detected (Compute {major}.{minor}) — using GPU mode.")
                use_gpu = True
            else:
                print(f"⚠️ GPU detected (Compute {major}.{minor}) — unsupported. Using CPU mode.")
        except Exception as e:
            print(f"⚠️ GPU query failed: {e}")
            use_gpu = False
    else:
        print("ℹ️ No CUDA device found — running on CPU.")

    try:
        reader = easyocr.Reader(['en'], gpu=use_gpu)
    except Exception as e:
        print(f"❌ GPU init failed ({e.__class__.__name__}: {e}) — retrying CPU.")
        reader = easyocr.Reader(['en'], gpu=False)
        use_gpu = False

    print(f"✅ OCR Engine ready → {'GPU' if use_gpu else 'CPU'} mode")
    return reader


# ========== SETTINGS ==========
INPUT_FOLDER = "TEMP/train_cars_cropped"
LABELS_CSV = "TEMP/train_car_labels.csv"
OUTPUT_FOLDER = "TEMP/train_cars_filtered"
CSV_OUTPUT = "TEMP/Train_car_IDs.csv"
REGION_FILE_MAIN = "Settings/car_ID_regions.json"
REGION_FILE_SECOND = "Settings/car_ID_regions_second_location.json"
RULES_FILE = "Settings/car_ID_rules.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with open(REGION_FILE_MAIN, 'r') as f:
    REGIONS_MAIN = json.load(f)
with open(REGION_FILE_SECOND, 'r') as f:
    REGIONS_SECOND = json.load(f)
with open(RULES_FILE, 'r') as f:
    RULES = json.load(f)

reader = init_easyocr()


# ========== LOAD LABELS ==========
def load_labels(csv_path):
    label_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["filename"].strip()] = row["label"].strip()
    print(f"✅ Loaded {len(label_map)} label entries.")
    return label_map

LABEL_MAP = load_labels(LABELS_CSV)


# ========== REGION HANDLING ==========
def extract_text_regions(image, label):
    h, w = image.shape[:2]
    regions_to_check = []

    if label in REGIONS_MAIN:
        regions_to_check.append(REGIONS_MAIN[label])
    elif "Unknown" in REGIONS_MAIN:
        regions_to_check.append(REGIONS_MAIN["Unknown"])

    if label in REGIONS_SECOND:
        regions_to_check.append(REGIONS_SECOND[label])

    crops = []
    for r in regions_to_check:
        x1, y1, x2, y2 = int(r["x1"] * w), int(r["y1"] * h), int(r["x2"] * w), int(r["y2"] * h)
        crops.append(image[y1:y2, x1:x2])
    return crops


def preprocess_image(img, allow_color=False):
    if not allow_color:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    return img


# ========== TEXT CLEANUP ==========
def clean_text(raw_text):
    """Normalize OCR output for consistent ID formatting."""
    text = re.sub(r'[^A-Za-z0-9]+', '', raw_text)
    return text


# ========== APPLY RULES ==========
def apply_car_rules(label, raw_texts):
    rule = RULES.get(label, RULES.get("Unknown", {}))
    regions_to_use = rule.get("regions_to_use", len(raw_texts))
    join_mode = rule.get("join_mode", "concat")
    combine_strategy = rule.get("combine_strategy", "all")

    selected_texts = raw_texts[:regions_to_use]

    if combine_strategy == "first":
        combined = selected_texts[0] if selected_texts else ""
    else:
        if join_mode == "underscore":
            combined = "_".join(selected_texts)
        else:
            combined = "".join(selected_texts)
    return clean_text(combined)


# ========== OCR PROCESS ==========
def process_car(filename, label):
    path = os.path.join(INPUT_FOLDER, filename)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {filename}")
        return ""

    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Failed to load: {filename}")
        return ""

    allow_color = RULES.get(label, {}).get("allow_color", False)
    crops = extract_text_regions(image, label)

    raw_texts = []
    for idx, crop in enumerate(crops):
        proc = preprocess_image(crop, allow_color)
        results = reader.readtext(proc)
        merged_text = " ".join([t[1] for t in results])
        raw_texts.append(merged_text)

        # Save overlay for debugging
        debug = crop.copy()
        for (bbox, text, conf) in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
            cv2.putText(debug, text, (pts[0][0], pts[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename[:-4]}_region{idx+1}.jpg"), debug)

    final_id = apply_car_rules(label, raw_texts)
    return final_id


# ========== MAIN ==========
def main():
    all_rows = []
    image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png'))])
    print(f"🟩 Processing {len(image_files)} cars...")

    for filename in image_files:
        label = LABEL_MAP.get(filename, "Unknown")
        print(f"🟩 {filename} → {label}")

        try:
            id_text = process_car(filename, label)
            all_rows.append([filename, label, id_text])
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")
            all_rows.append([filename, label, "" ])

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "Car_Type", "ID"])
        writer.writerows(all_rows)

    print(f"✅ Finished → {CSV_OUTPUT}")
    print(f"🖼️ Overlays saved → {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
