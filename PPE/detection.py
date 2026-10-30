#!/usr/bin/env python3
import math
import queue
import sys
import threading
from pathlib import Path
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from hailo_platform import HEF

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# The subpackages import their siblings by bare module name (for example
# `from scrfd_postproc import *`), so their directories must be importable too.
for _subpackage in ("face_detection", "goggle_detection"):
    _subpackage_path = str(SCRIPT_DIR / _subpackage)
    if _subpackage_path not in sys.path:
        sys.path.append(_subpackage_path)

from common.tools import init_rpicam2
from common.utils import letterbox_img
from face_detection import scrfd_detection as scrfd
from goggle_detection import goggle_detection as goggles
from goggle_detection.goggle_post_process import draw_detections, extract_detections

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

SCRFD_HEF_PATH = SCRIPT_DIR / "face_detection" / "scrfd_500m.hef"
GOGGLE_HEF_PATH = SCRIPT_DIR / "goggle_detection" / "matrix_f26_best_10232026.hef"

WINDOW_NAME = "Face Detections"
TILE_SIZE = 256
QUEUE_TIMEOUT_S = 0.5
THREAD_JOIN_TIMEOUT_S = 2

@dataclass
class GoggleState:
    """
    Represents the state of a subject for which we are detecting goggles.
    track_id: ByteTrack ID for the subject
    goggle_detection_amount: Amount of times goggles were detected
    person_detection_amount: Amount of times the subject was detected
    frames_since_last_face: Amount of frames input since last time this subject was seen
    """
    track_id: int
    goggle_detection_amount: int
    person_detection_amount: int
    frames_since_last_face: int
    

def run_goggle_pipeline(input_queue: queue.Queue[scrfd.SCRFDOutput],
                        goggle_input_queue: queue.Queue,
                        goggle_output_queue: queue.Queue,
                        output_queue: queue.Queue,
                        size: int,
                        config_data: dict,
                        stop_event: threading.Event) -> None:
    """
    Runs the goggles model on every face crop produced by the SCRFD pipeline.

    Crops are letterboxed to the model input size and pushed through
    goggle_detection.run_inference_pipeline (which owns the Hailo device). Each result
    comes back paired with the exact crop it was computed from, so no matching by index
    is needed.

    Params:
        input_queue: Queue of postprocess outputs from SCRFD, each containing "faces".
        goggle_input_queue: Queue consumed by goggle_detection.run_inference_pipeline.
        goggle_output_queue: Queue produced by goggle_detection.run_inference_pipeline.
        output_queue: Queue for the SCRFD output augmented with "face_detections", a
            list of (face crop, detections dict) pairs.
        model_input_width: Goggles model input width.
        model_input_height: Goggles model input height.
        config_data: Parsed config.json used by extract_detections.
        stop_event: threading.Event to stop this pipeline.
    """
    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=QUEUE_TIMEOUT_S)
        except queue.Empty:
            continue

        # goggle_detection feeds camera frames to the model without a colour
        # conversion, so the crops are passed through the same way.
        faces = [face for face in input.faces if face.size > 0]
        for face in faces:
            goggle_input_queue.put((face, letterbox_img(face)))

        face_detections = []
        while len(face_detections) < len(faces) and not stop_event.is_set():
            try:
                face, raw_inference = goggle_output_queue.get(timeout=QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue
            face_detections.append(
                (face, extract_detections(face, raw_inference, config_data))
            )

        output = asdict(input)
        output["face_detections"] = face_detections
        output_queue.put(output)


def render_face_grid(face_detections: list, labels: list) -> np.ndarray:
    """
    Builds a grid of face crops with the goggles model's boxes drawn on each one.

    Params:
        face_detections: List of (face crop, detections dict) pairs.
        labels: Class labels used for box captions.
    Returns:
        BGR image suitable for cv2.imshow.
    """
    if not face_detections:
        canvas = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        cv2.putText(canvas, "No faces", (16, TILE_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    cols = max(1, math.ceil(math.sqrt(len(face_detections))))
    rows = math.ceil(len(face_detections) / cols)
    canvas = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE, 3), dtype=np.uint8)

    for i, (face, detections) in enumerate(face_detections):
        # Stretch the crop into the square tile, matching the model's stretched input.
        height, width = face.shape[:2]
        scale_x = TILE_SIZE / width
        scale_y = TILE_SIZE / height

        tile = cv2.resize(face, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)

        draw_detections(
            {
                "detection_boxes": [
                    [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                    for x1, y1, x2, y2 in detections["detection_boxes"]
                ],
                "detection_scores": detections["detection_scores"],
                "detection_classes": detections["detection_classes"],
                "num_detections": detections["num_detections"],
            },
            tile,
            labels,
        )

        row, col = divmod(i, cols)
        canvas[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE] = tile

    return canvas


def main() -> None:
    """Runs the SCRFD pipeline followed by the goggles model, and displays the faces."""
    camera = init_rpicam2(CAMERA_WIDTH, CAMERA_HEIGHT)
    assert camera is not None, "Failed to open the Raspberry Pi camera."

    labels = goggles.get_labels(str(goggles.DEFAULT_LABELS_PATH))
    config_data = goggles.load_json_file(goggles.DEFAULT_CONFIG_PATH)

    goggle_hef = HEF(str(GOGGLE_HEF_PATH))
    model_input_height, model_input_width, _ = goggle_hef.get_input_vstream_infos()[0].shape
    assert model_input_width == model_input_height, "We only support models with square input widths/heights"

    stop_event = threading.Event()

    input_queue = queue.Queue()
    preprocessed_frames = queue.Queue()
    inferences = queue.Queue()
    face_outputs = queue.Queue()
    goggle_inputs = queue.Queue()
    goggle_outputs = queue.Queue()
    outputs = queue.Queue()

    threads = [
        threading.Thread(
            target=scrfd.run_preprocess_pipeline,
            args=(input_queue, preprocessed_frames, stop_event),
        ),
        threading.Thread(
            target=scrfd.run_inference_pipeline,
            args=(str(SCRFD_HEF_PATH), preprocessed_frames, face_outputs, stop_event),
        ),
        # Owns the second VDevice, so it runs in its own thread alongside SCRFD.
        threading.Thread(
            target=run_goggle_pipeline,
            args=(face_outputs, goggle_inputs, goggle_outputs, outputs,
                  model_input_height, config_data, stop_event),
        ),
        threading.Thread(
            target=goggles.run_inference_pipeline,
            args=(str(GOGGLE_HEF_PATH), goggle_inputs, goggle_outputs, stop_event),
        )
    ]

    for thread in threads:
        thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = camera.read()
            if ok:
                input_queue.put(frame)

            # Drain the queue so we always display the most recent result.
            latest = None
            while True:
                try:
                    latest = outputs.get_nowait()
                except queue.Empty:
                    break

            if latest is not None:
                cv2.imshow(WINDOW_NAME, render_face_grid(latest["face_detections"], labels))

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=THREAD_JOIN_TIMEOUT_S)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
