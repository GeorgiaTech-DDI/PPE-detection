#!/usr/bin/env python3
import math
import queue
import sys
import threading
import time
from pathlib import Path
from dataclasses import dataclass

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
from goggle_state import GoggleState, GoggleStateTracker, WearState, score_frame

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

SCRFD_HEF_PATH = SCRIPT_DIR / "face_detection" / "scrfd_500m.hef"
GOGGLE_HEF_PATH = SCRIPT_DIR / "goggle_detection" / "matrix_f26_best_10232026.hef"

WINDOW_NAME = "Face Detections"
TILE_SIZE = 256
QUEUE_TIMEOUT_S = 0.5
THREAD_JOIN_TIMEOUT_S = 2

SAFETY_GLASSES_LABEL = "safety_glasses"
# Matches how ByteTrack converts lost_track_buffer into frames before dropping a track.
EVICT_AFTER_FRAMES = int(scrfd.FRAME_RATE / 30.0 * scrfd.LOST_TRACK_BUFFER)

STATE_COLORS = {
    WearState.UNKNOWN: (160, 160, 160),
    WearState.WEARING: (0, 200, 0),
    WearState.NOT_WEARING: (0, 0, 220),
}

@dataclass
class GoggleOutput:
    """
    Goggle detections for every tracked face in one frame.
    timestamp: time.monotonic() when the frame reached the goggle pipeline
    face_detections: Track ID mapped to (face crop, extract_detections output)
    """
    timestamp: float
    face_detections: dict[int, tuple[np.ndarray, dict]]


def run_goggle_pipeline(input_queue: queue.Queue[scrfd.SCRFDOutput],
                        goggle_input_queue: queue.Queue,
                        goggle_output_queue: queue.Queue,
                        output_queue: queue.Queue[GoggleOutput],
                        size: int,
                        config_data: dict,
                        stop_event: threading.Event) -> None:
    """
    Runs the goggles model on every tracked face crop produced by the SCRFD pipeline.

    Crops are letterboxed to the model input size and pushed through
    goggle_detection.run_inference_pipeline (which owns the Hailo device). Each result
    comes back paired with the track ID and crop it was computed from, so no matching
    by index is needed.

    Params:
        input_queue: Queue of postprocess outputs from SCRFD, each containing "faces_by_id".
        goggle_input_queue: Queue consumed by goggle_detection.run_inference_pipeline.
        goggle_output_queue: Queue produced by goggle_detection.run_inference_pipeline.
        output_queue: Queue of GoggleOutput, one per SCRFD frame (including frames
            without faces, so frame counts stay in step with ByteTrack).
        size: Goggles model input size.
        config_data: Parsed config.json used by extract_detections.
        stop_event: threading.Event to stop this pipeline.
    """
    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=QUEUE_TIMEOUT_S)
        except queue.Empty:
            continue
        timestamp = time.monotonic()

        # goggle_detection feeds camera frames to the model without a colour
        # conversion, so the crops are passed through the same way. It hands the
        # first tuple element back untouched, so it carries the track ID along.
        faces_by_id = {
            track_id: face
            for track_id, face in (input.faces_by_id or {}).items()
            if face.size > 0
        }
        for track_id, face in faces_by_id.items():
            goggle_input_queue.put(((track_id, face), letterbox_img(face)))

        face_detections = {}
        while len(face_detections) < len(faces_by_id) and not stop_event.is_set():
            try:
                (track_id, face), raw_inference = goggle_output_queue.get(timeout=QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue
            face_detections[track_id] = (
                face, extract_detections(face, raw_inference, config_data)
            )

        output_queue.put(GoggleOutput(timestamp=timestamp, face_detections=face_detections))


def render_face_grid(face_detections: dict[int, tuple[np.ndarray, dict]],
                     states: dict[int, GoggleState],
                     labels: list) -> np.ndarray:
    """
    Builds a grid of face crops with the goggles model's boxes drawn on each one,
    captioned with each track's ID and goggle state.

    Params:
        face_detections: Track ID mapped to (face crop, detections dict).
        states: GoggleStateTracker states, keyed by track ID.
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

    for i, (track_id, (face, detections)) in enumerate(face_detections.items()):
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

        goggle_state = states.get(track_id)
        if goggle_state is not None:
            color = STATE_COLORS[goggle_state.state]
            score = goggle_state.window_score
            caption = f"#{track_id} {goggle_state.state.value}"
            if score is not None:
                caption += f" {score:.2f}"
            cv2.rectangle(tile, (0, 0), (TILE_SIZE - 1, TILE_SIZE - 1), color, 4)
            cv2.putText(tile, caption, (8, TILE_SIZE - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

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

    safety_glasses_class = labels.index(SAFETY_GLASSES_LABEL)
    state_tracker = GoggleStateTracker(evict_after_frames=EVICT_AFTER_FRAMES)

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
            kwargs={"track_only_within_1m": True},
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

            # Drain the queue so we always display the most recent result. Every output
            # still goes through the state tracker so no frames are skipped.
            latest = None
            while True:
                try:
                    latest = outputs.get_nowait()
                except queue.Empty:
                    break
                state_tracker.update(
                    latest.timestamp,
                    {
                        track_id: score_frame(detections, safety_glasses_class)
                        for track_id, (_, detections) in latest.face_detections.items()
                    },
                )

            if latest is not None:
                cv2.imshow(WINDOW_NAME, render_face_grid(
                    latest.face_detections, state_tracker.states, labels
                ))

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
