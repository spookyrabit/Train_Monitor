import shutil
from pathlib import Path

# Define the TEMP folder path
TEMP_FOLDER = Path("TEMP")

# Delete TEMP if it exists
if TEMP_FOLDER.exists() and TEMP_FOLDER.is_dir():
    shutil.rmtree(TEMP_FOLDER)
    print(f"Deleted existing folder: {TEMP_FOLDER}")

# Recreate TEMP folder
TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"Created new folder: {TEMP_FOLDER}")
