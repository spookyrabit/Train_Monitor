import os
import json
import csv
import cv2
import torch
import numpy as np
import easyocr
import re
import sys

# =======================================
# GPU AUTO-FALLBACK HANDLER
# =======================================
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

# =======================================
# SETTINGS
# =======================================
INPUT_FOLDER = "TEMP/train_cars_cropped"
LABELS_CSV = "TEMP/train_car_labels.csv"
OUTPUT_FOLDER = "TEMP/train_cars_filtered"
CSV_OUTPUT = "TEMP/Train_car_IDs.csv"

REGION_FILE_MAIN = "Settings/car_ID_regions.json"
REGION_FILE_SECOND = "Settings/car_ID_regions_second_location.json"
RULES_FILE = "Settings/car_ID_rules.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("Settings", exist_ok=True)

# =======================================
# SAFE LOAD JSON HELPERS
# =======================================
def safe_load_json(path, default):
    if not os.path.exists(path):
        print(f"[WARN] Missing {path} — creating default.")
        with open(path, "w") as f:
            json.dump(default, f, indent=2)
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e} — using default.")
        return default

# Default fallbacks if settings are missing
REGIONS_MAIN = safe_load_json(REGION_FILE_MAIN, {"Unknown": {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6}})
REGIONS_SECOND = safe_load_json(REGION_FILE_SECOND, {})
RULES = safe_load_json(RULES_FILE, {"Unknown": {"regions_to_use": 1, "join_mode": "concat", "combine_strategy": "first"}})

reader = init_easyocr()

# =======================================
# LOAD LABELS (SAFE)
# =======================================
def load_labels(csv_path):
    label_map = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] Label file missing: {csv_path}. Creating placeholder.")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            f.write("filename,label,confidence,height,height_category\n")
        return label_map

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "").strip()
                label = row.get("label", "").strip() or "Unknown"
                if fname:
                    label_map[fname] = label
        print(f"✅ Loaded {len(label_map)} label entries.")
    except Exception as e:
        print(f"[ERROR] Failed to read labels CSV: {e}")
    return label_map

LABEL_MAP = load_labels(LABELS_CSV)

# =======================================
# REGION HANDLING
# =======================================
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
        try:
            x1, y1, x2, y2 = int(r["x1"] * w), int(r["y1"] * h), int(r["x2"] * w), int(r["y2"] * h)
            crops.append(image[y1:y2, x1:x2])
        except Exception as e:
            print(f"[WARN] Bad region for {label}: {e}")
    return crops if crops else [image]  # fallback to whole image

# =======================================
# IMAGE PREPROCESSING
# =======================================
def preprocess_image(img, allow_color=False):
    if not allow_color:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    return img

# =======================================
# TEXT CLEANUP
# =======================================
def clean_text(raw_text):
    text = re.sub(r'[^A-Za-z0-9]+', '', raw_text)
    return text.strip()

# =======================================
# APPLY RULES
# =======================================
def apply_car_rules(label, raw_texts):
    rule = RULES.get(label, RULES.get("Unknown", {}))
    regions_to_use = rule.get("regions_to_use", len(raw_texts))
    join_mode = rule.get("join_mode", "concat")
    combine_strategy = rule.get("combine_strategy", "all")

    selected_texts = raw_texts[:regions_to_use]
    if not selected_texts:
        return ""

    if combine_strategy == "first":
        combined = selected_texts[0]
    else:
        combined = "_".join(selected_texts) if join_mode == "underscore" else "".join(selected_texts)

    return clean_text(combined)

# =======================================
# OCR PROCESS
# =======================================
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
        try:
            proc = preprocess_image(crop, allow_color)
            results = reader.readtext(proc)
            merged_text = " ".join([t[1] for t in results])
            raw_texts.append(merged_text)

            # Debug overlay
            debug = crop.copy()
            for (bbox, text, conf) in results:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
                cv2.putText(debug, text, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename[:-4]}_region{idx+1}.jpg"), debug)
        except Exception as e:
            print(f"[ERROR] OCR failure on {filename} region {idx+1}: {e}")

    final_id = apply_car_rules(label, raw_texts)
    return final_id or "UNKNOWN"

# =======================================
# MAIN LOOP
# =======================================
def main():
    all_rows = []
    try:
        image_files = sorted([f for f in os.listdir(INPUT_FOLDER)
                              if f.lower().endswith(('.jpg', '.png'))])
    except Exception as e:
        print(f"[FATAL] Could not list images: {e}")
        image_files = []

    print(f"🟩 Processing {len(image_files)} cars...")

    for filename in image_files:
        label = LABEL_MAP.get(filename, "Unknown")
        print(f"🟩 {filename} → {label}")

        try:
            id_text = process_car(filename, label)
            all_rows.append([filename, label, id_text])
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")
            all_rows.append([filename, label, "UNKNOWN"])

    # Always write a valid CSV
    try:
        with open(CSV_OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "Car_Type", "ID"])
            writer.writerows(all_rows)
        print(f"✅ Finished → {CSV_OUTPUT}")
    except Exception as e:
        print(f"[FATAL] Could not write output CSV: {e}")
        os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
        with open(CSV_OUTPUT, "w", newline="") as f:
            f.write("filename,Car_Type,ID\n")
        print(f"[INFO] Created empty fallback file → {CSV_OUTPUT}")

    print(f"🖼️ Overlays saved → {OUTPUT_FOLDER}")
    print("[INFO] Read_IDs.py finished (fail-safe mode)")

if __name__ == "__main__":
    main()
