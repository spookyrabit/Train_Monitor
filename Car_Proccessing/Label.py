import os
from PIL import Image
import torch
import clip
from tqdm import tqdm
import csv

# -----------------------------
# Settings
# -----------------------------
INPUT_FOLDER = "TEMP/train_cars_cropped"  # folder with height-cropped train car images
OUTPUT_CSV = "TEMP/train_car_labels.csv"
DEVICE = "cpu"  # CPU-only mode

# Height threshold for Short vs Tall
HEIGHT_THRESHOLD = 500  # pixels; above = Tall, below = Short

# Candidate labels and multiple prompts per type
CANDIDATE_LABELS = {
    "Diesel Engine": [
        "short engine car, diesel locomotive, front of train",
        "diesel engine, short metal vehicle, railway engine",
        "train diesel locomotive, engine at front of train"
    ],
    "Steam Engine": [
        "steam locomotive, front engine, vintage train",
        "old steam train engine, front of train",
        "steam railway engine, black metal front car"
    ],
    "Boxcar": [
        "tall rectangular freight boxcar, metal siding",
        "tall boxcar with sliding doors, cargo train car",
        "standard tall boxcar, railway freight car",
        "high-profile boxcar, large cargo car"
    ],
    "Short Boxcar": [
        "short rectangular boxcar, small freight car",
        "short boxcar for tunnels, compact freight",
        "low-profile boxcar, railway short cargo car"
    ],
    "Hopper": [
        "tall trapezoidal open-top hopper car, carries sand or ore",
        "freight hopper car, tall angled sides",
        "tall open hopper, railway cargo car",
        "White/Tan V shaped rail hopper car"
    ],
    "Passenger": [
        "short rectangular passenger car, windows for people",
        "passenger train car, short, transport people",
        "railway coach, short train car, human transport"
    ]
}

# -----------------------------
# Load CLIP
# -----------------------------
print("[INFO] Loading CLIP model on CPU...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
print("[INFO] Model loaded successfully.")

# -----------------------------
# Helper functions
# -----------------------------
def classify_image_with_height(image_path, height_category):
    """
    Classify a single train car image using CLIP with height-informed candidate labels.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Select candidate types based on height category
    if height_category == "Short":
        candidate_types = ["Short Boxcar", "Passenger", "Diesel Engine", "Steam Engine"]
    else:
        candidate_types = ["Boxcar", "Hopper", "Diesel Engine", "Steam Engine"]

    # Build text prompts for CLIP
    text_prompts = []
    label_mapping = []
    for label in candidate_types:
        prompts = CANDIDATE_LABELS[label]
        text_prompts.extend(prompts)
        label_mapping.extend([label] * len(prompts))

    # Encode
    text_tokens = clip.tokenize(text_prompts).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (100.0 * image_features @ text_features.T).squeeze(0)
    best_idx = similarity.argmax().item()
    confidence = similarity[best_idx].item() / 100.0
    return label_mapping[best_idx], confidence, image.height

def infer_engines_bidirectional(results):
    """
    Identify candidate engines at either end of the train using relative heights.
    """
    # If all cars are short, engines likely at either end
    heights = [r["height"] for r in results]
    median_height = sorted(heights)[len(heights)//2]

    # Start end
    for r in results:
        if r["height"] < median_height * 0.9:
            r["label"] = "Diesel Engine"
        else:
            break

    # End
    for r in reversed(results):
        if r["height"] < median_height * 0.9:
            r["label"] = "Diesel Engine"
        else:
            break

    return results

# -----------------------------
# Main processing loop
# -----------------------------
images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
images.sort()  # maintain order

results = []

for img_name in tqdm(images, desc="Classifying"):
    img_path = os.path.join(INPUT_FOLDER, img_name)
    img_height = Image.open(img_path).height

    # Determine height category using HEIGHT_THRESHOLD
    height_category = "Tall" if img_height > HEIGHT_THRESHOLD else "Short"

    label, conf, height = classify_image_with_height(img_path, height_category)
    results.append({
        "filename": img_name,
        "label": label,
        "confidence": conf,
        "height": height,
        "height_category": height_category
    })
    print(f"{img_name}: label={label}, conf={conf:.3f}, height={height}px, category={height_category}")

# -----------------------------
# Infer engines at train ends
# -----------------------------
results = infer_engines_bidirectional(results)

# -----------------------------
# Save results
# -----------------------------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = ["filename", "label", "confidence", "height", "height_category"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"[INFO] Classification complete → {OUTPUT_CSV}")
