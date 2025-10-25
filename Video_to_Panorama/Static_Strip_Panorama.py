import cv2
import numpy as np

video_path = "train.mp4"
cap = cv2.VideoCapture(video_path)

# Grab video dimensions
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")
height, width, _ = frame.shape

# Pick the column to sample
column_x = width // 2
panorama = []

while ret:
    column = frame[:, column_x:column_x+25, :]  # 1-pixel-wide vertical strip
    panorama.append(column)
    ret, frame = cap.read()

# Stack all columns side-by-side
panorama_img = np.concatenate(panorama, axis=1)
cv2.imwrite("TEMP/train_panorama.jpg", panorama_img)
cap.release()
