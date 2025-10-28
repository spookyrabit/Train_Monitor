# run_range_once.py
import subprocess
import os
from datetime import datetime

RANGE_SCRIPTS = [
    r"Prep/Choose_Video.py",
    r"Prep/Clear_Folder.py",
    r"Prep/Align_Video.py",
    r"Video_to_Panorama/Rectangle_Tracking_Panorama.py",
    r"Image_Prep/Split_Panorama.py",
    r"Image_Prep/Crop_To_Car_Height.py",
    r"Car_Proccessing/Label.py",
    r"Car_Proccessing/Read_IDs.py",
    r"Prep/Save_Train_Info.py"
 
]

def run_script(path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Running: {path}")
    try:
        subprocess.run([
            "python",
            path
        ], check=True, cwd=os.path.dirname(os.path.abspath(__file__)).split("Cycles")[0])
        print(f"[{ts}] Completed: {path}")
    except subprocess.CalledProcessError as e:
        print(f"[{ts}] Error ({path}): {e}")

def main():
    for script in RANGE_SCRIPTS:
        run_script(script)

if __name__ == "__main__":
    main()
