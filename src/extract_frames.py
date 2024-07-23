import cv2
import os


def extract_frames(video_path, output_folder, frame_skip=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved_count = 0

    while success:
        if count % frame_skip == 0:
            frame_name = f"frame{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1

    print(f"Extracted {saved_count} frames.")
