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
    conf_threshold = 0.2  # Установите порог уверенности на нужное значение

    model = load_model(model_name)
    detect_anomalies(model, frame_folder, output_folder, conf_threshold)


# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed

# logging.basicConfig(
#      level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def load_model(model_name='yolov8n.pt'):
#     models_dir = os.path.join(
#         os.path.dirname(os.path.dirname(__file__)), 'models')
#     model_path = os.path.join(models_dir, model_name)
#     if not os.path.exists(model_path):
#         model = YOLO(model_name)
#         model.save(model_path)
#     else:
#         model = YOLO(model_path)

#     if hasattr(model, 'fuse'):
#         try:
#             model.fuse()
#         except AttributeError as e:
#             logging.warning(f"Fuse method failed: {e}")

#     return model


# def calculate_frame_difference(prev_frame, curr_frame, threshold=10):
#     diff = cv2.absdiff(prev_frame, curr_frame)
#     _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
#     return np.sum(thresh) / thresh.size


# def process_frame(model, frame_path,
#                   prev_frame, prev_objects,
#                   diff_threshold, obj_threshold):
#     curr_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

#     # Calculate frame difference
#     diff_percent = calculate_frame_difference(
#         prev_frame, curr_frame) if prev_frame is not None else 0

#     # Detect objects
#     results = model(frame_path)[0]
#     curr_objects = results.boxes.data.cpu().numpy()

#     is_anomaly = False

#     if diff_percent > diff_threshold:
#         is_anomaly = True
#         logging.info(f"{os.path.basename(frame_path)}: Anomaly detected"
#                      " - High pixel change: {diff_percent:.2%}")

#     if prev_objects is not None and len(curr_objects) != len(prev_objects):
#         is_anomaly = True
#         logging.info(f"{os.path.basename(frame_path)}: Anomaly detected"
#                      " - Change in object count: {len(curr_objects)}")

#     return is_anomaly, curr_frame, curr_objects


# def detect_anomalies(model, frames_folder,
#                      output_folder, diff_threshold=0.02, obj_threshold=0.2):
#     frames = sorted(
#         [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
#         )
#     prev_frame = None
#     prev_objects = None

#     with ThreadPoolExecutor() as executor:
#         future_to_frame = {executor.submit(
#             process_frame, model, os.path.join(
#                 frames_folder, frame),
#             prev_frame, prev_objects, diff_threshold, obj_threshold
#                 ): frame for frame in frames}

#         for future in as_completed(future_to_frame):
#             frame = future_to_frame[future]
#             try:
#                 is_anomaly, curr_frame, curr_objects = future.result()

#                 if is_anomaly:
#                     output_path = os.path.join(
#                         output_folder, f"anomaly_{frame}")
#                     cv2.imwrite(output_path, cv2.imread(
#                         os.path.join(frames_folder, frame)))

#                 prev_frame = curr_frame
#                 prev_objects = curr_objects

#             except Exception as exc:
#                 logging.error(f'{frame} generated an exception: {exc}')

#     logging.info("Anomaly detection completed.")


# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_name = 'yolov8n.pt'
#     frame_folder = os.path.join(base_dir, '../frames')
#     output_folder = os.path.join(base_dir, '../output')
#     diff_threshold = 0.02
#     obj_threshold = 0.2

#     model = load_model(model_name)
#     detect_anomalies(model, frame_folder,
#                      output_folder,
#                      diff_threshold,
#                      obj_threshold)


# import os
# import logging
# import concurrent.futures
# from ultralytics import YOLO


# logging.basicConfig(
#     level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def load_model(model_name='yolov8n.pt'):
#     models_dir = os.path.join(
#         os.path.dirname(os.path.dirname(__file__)), 'models')
#     model_path = os.path.join(models_dir, model_name)
#     if not os.path.exists(model_path):
#         model = YOLO(model_name)
#         model.save(model_path)
#     else:
#         model = YOLO(model_path)

#     if hasattr(model, 'fuse'):
#         try:
#             model.fuse()
#         except AttributeError as e:
#             logging.warning(f"Fuse method failed: {e}")

#     return model


# def process_frame(frame_path, model, output_folder,
#                   conf_threshold, prev_detections):
#     try:
#         results = model.predict(frame_path, conf=conf_threshold)
#         detections = results[0].boxes.data

#         current_detections = len(detections)
#         # Detect significant changes in object count
#         is_anomaly = abs(current_detections - prev_detections) > 1

#         if is_anomaly:
#             result_path = os.path.join(
#                 output_folder, f'result_{os.path.basename(frame_path)}')
#             results[0].save(result_path)
#             return (frame_path, current_detections, True)
#         else:
#             return (frame_path, current_detections, False)
#     except Exception as e:
#         logging.error(f"Error processing frame {frame_path}: {e}")
#         return (frame_path, -1, False)


# def detect_anomalies(model, frame_folder, output_folder, conf_threshold=0.3):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     frame_files = sorted(
#         [os.path.join(frame_folder, f) for f in os.listdir(
#             frame_folder) if f.endswith('.jpg')]
#             )

#     logging.info(f"Starting anomaly detection on {len(frame_files)} frames")

#     prev_detections = 0
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         for frame in frame_files:
#             future = executor.submit(process_frame, frame, model,
#                                      output_folder, conf_threshold,
#                                      prev_detections)
#             frame_file, num_detections, is_anomaly = future.result()

#             if is_anomaly:
#                 logging.info(f'{os.path.basename(frame_file)}'
#                             ' - Anomaly detected! Objects: {num_detections}')
#             else:
#                 logging.info(
#                     f'{os.path.basename(frame_file)}'
#                     ' - No anomalies detected. Objects: {num_detections}')

#             prev_detections = num_detections


# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_name = 'yolov8n.pt'  # Update with your model path
#     frame_folder = os.path.join(base_dir, '../frames')
#     output_folder = os.path.join(base_dir, '../output')
#     conf_threshold = 0.3  # Increased confidence threshold

#     model = load_model(model_name)
#     detect_anomalies(model, frame_folder, output_folder, conf_threshold)
