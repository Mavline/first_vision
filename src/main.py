import os
import logging
from extract_frames import extract_frames
from detect_anomalies import load_model, detect_anomalies

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(video_path, model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frames_folder = os.path.join(base_dir, '../frames')
    output_folder = os.path.join(base_dir, '../output')
    frame_skip = 30  # Adjust as needed

    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logging.info("Starting frame extraction")
    extract_frames(video_path, frames_folder, frame_skip)

    logging.info("Starting anomaly detection")
    model = load_model(model_name)
    detect_anomalies(model, frames_folder, output_folder)

    logging.info("Processing completed.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, '../data/sample_video.mp4')
    model_name = 'yolov8n.pt'
    main(video_path, model_name)
