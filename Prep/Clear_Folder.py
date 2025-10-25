import shutil
from pathlib import Path

# Define base directories
BASE_PATH = Path("/home/spooky/Documents/Train_Monitor")
DATABASE_PATH = BASE_PATH / "Database"
TEMP_FOLDER = BASE_PATH / "TEMP"

# 1. Delete TEMP if it exists
if TEMP_FOLDER.exists() and TEMP_FOLDER.is_dir():
    shutil.rmtree(TEMP_FOLDER)
    print(f"Deleted existing folder: {TEMP_FOLDER}")


# 2. Recreate TEMP folder
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"Created new folder: {TEMP_FOLDER}")


# 3. Find the video that was copied (should only be one .mp4 file)
videos = list(BASE_PATH.glob("*.mp4"))
if not videos:
    print("No video found in Train_Monitor directory.")
    exit(1)

if len(videos) > 1:
    print("Warning: Multiple videos found. Using the first one found.")
    
video_path = videos[0]
video_name = video_path.stem  # e.g. "10-22-2025 13:22:00"
print(f"Processing video: {video_path.name}")

# 4. Create the corresponding Database folder
new_db_folder = DATABASE_PATH / video_name
new_db_folder.mkdir(parents=True, exist_ok=True)
print(f"Created database folder: {new_db_folder}")

# 5. Rename the video to 'train.mp4'
new_video_path = BASE_PATH / "train.mp4"
video_path.rename(new_video_path)
print(f"Renamed video to: {new_video_path.name}")

print("Setup complete. TEMP folder and Database structure ready.")
