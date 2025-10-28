import os
import shutil

# Path to the text file that tells us the folder name
txt_file_path = "TEMP/Video_Name.txt"

# List of files we want to move
files_to_move = ["TEMP/train_car_labels.csv",
                 "TEMP/Train_car_IDs.csv"
                 ]  # Add more file paths here if needed

# Read the folder name from the text file
with open(txt_file_path, "r") as f:
    folder_name = f.read().strip()  # "10-25-2025 14:10:00"

# Create the target folder if it doesn't exist
target_folder = os.path.join("Database", folder_name)
os.makedirs(target_folder, exist_ok=True)

# Move each file to the target folder
for file_path in files_to_move:
    if os.path.exists(file_path):
        shutil.move(file_path, target_folder)
        print(f"Moved '{file_path}' to '{target_folder}'")
    else:
        print(f"File '{file_path}' does not exist. Skipping.")
