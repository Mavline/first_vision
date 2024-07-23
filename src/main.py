import glob
import os
import logging
from extract_frames import extract_frames
from detect_anomalies import load_model, detect_anomalies

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_directories(base_dir):
    folders = ['data', 'frames', 'output', 'models']
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            open(os.path.join(folder_path, '.gitkeep'), 'w').close()
            logging.info(f"Created directory: {folder_path}")


def main(video_path, model_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(base_dir, 'data')
    video_files = glob.glob(os.path.join(data_folder, '*.mp4'))

    if not video_files:
        logging.error("No .mp4 files found in the data folder")
        return

    video_path = video_files[0]  # Берем первый найденный .mp4 файл
    ensure_directories(base_dir)

    frames_folder = os.path.join(base_dir, 'frames')
    output_folder = os.path.join(base_dir, 'output')
    frame_skip = 30  # Adjust as needed

    logging.info("Starting frame extraction")
    extract_frames(video_path, frames_folder, frame_skip)

    logging.info("Starting anomaly detection")
    model = load_model(model_name)
    detect_anomalies(model, frames_folder, output_folder)

    logging.info("Processing completed.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, 'data', 'sample_video.mp4')
    model_name = 'yolov8n.pt'
    main(video_path, model_name)
