import os
import shutil

def find_and_copy_missing_video():
    base_path = "/home/spooky/Documents/Train_Monitor"
    videos_path = os.path.join(base_path, "Videos")
    database_path = os.path.join(base_path, "Database")

    # Iterate through video files in Videos directory
    for filename in sorted(os.listdir(videos_path)):
        if filename.endswith(".mp4"):
            video_name = os.path.splitext(filename)[0]
            video_path = os.path.join(videos_path, filename)
            corresponding_folder = os.path.join(database_path, video_name)

            # Check if the corresponding folder exists
            if not os.path.isdir(corresponding_folder):
                # Folder missing → copy video to base directory
                destination = os.path.join(base_path, filename)
                shutil.copy2(video_path, destination)
                print(f"Copied '{filename}' to '{destination}' (no folder found).")
                return  # Stop after handling one video

    print("All videos have corresponding folders.")

if __name__ == "__main__":
    find_and_copy_missing_video()
