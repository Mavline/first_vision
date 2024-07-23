import os
import logging
import concurrent.futures
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name='yolov8n.pt'):
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        model = YOLO(model_name)
        model.save(model_path)
    else:
        model = YOLO(model_path)

    if hasattr(model, 'fuse'):
        try:
            model.fuse()
        except AttributeError as e:
            logging.warning(f"Fuse method failed: {e}")

    return model


def process_frame(frame_path, model, output_folder, conf_threshold):
    try:
        results = model.predict(frame_path, conf=conf_threshold)
        detections = results[0].boxes.data

        if len(detections) > 0:
            result_path = os.path.join(
                output_folder, f'result_{os.path.basename(frame_path)}')
            results[0].save(result_path)
            return (frame_path, len(detections))
        else:
            return (frame_path, 0)
    except Exception as e:
        logging.error(f"Error processing frame {frame_path}: {e}")
        return (frame_path, -1)


def detect_anomalies(model, frame_folder, output_folder, conf_threshold=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_files = sorted(
        [os.path.join(frame_folder, f) for f in os.listdir(
            frame_folder) if f.endswith('.jpg')])

    logging.info(f"Starting anomaly detection on {len(frame_files)} frames")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(
            process_frame, frame, model, output_folder, conf_threshold
            ) for frame in frame_files]
        for future in concurrent.futures.as_completed(futures):
            frame_file, num_detections = future.result()
            if num_detections > 0:
                logging.info(
                    f'{os.path.basename(frame_file)} -'
                    f'Anomalies: {num_detections}')
            elif num_detections == 0:
                logging.info(
                    f'{os.path.basename(frame_file)} - '
                    'No anomalies detected')
            else:
                logging.error(
                    f'{os.path.basename(frame_file)} - '
                    'Error processing frame')


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = 'yolov8n.pt'  # Update with your model path
    frame_folder = os.path.join(base_dir, '../frames')
    output_folder = os.path.join(base_dir, '../output')
    conf_threshold = 0.5  # Установите порог уверенности на нужное значение

    model = load_model(model_name)
    detect_anomalies(model, frame_folder, output_folder, conf_threshold)
