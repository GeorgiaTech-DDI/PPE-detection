#!/usr/bin/env python3
import json
import logging
import queue
import sys
import threading
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    FormatType,
    HEF,
    HailoSchedulingAlgorithm,
    VDevice,
)

SCRIPT_DIR = Path(__file__).resolve().parent
# Make `common` importable when running the script from any directory.
sys.path.insert(0, str(SCRIPT_DIR.parent))

from common.utils import letterbox_img
from common.tools import init_rpicam2
from goggle_argparse import parse_args
from frame_rate_tracker import FrameRateTracker
from goggle_post_process import inference_result_handler

DEFAULT_HEF_PATH = SCRIPT_DIR / "best_mask_and_goggles.hef"
DEFAULT_LABELS_PATH = SCRIPT_DIR / "goggle_labels.txt"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
CAMERA_SOURCE_FPS = 30.0

TIMEOUT_MS = 10000

MAX_INPUT_QUEUE_SIZE = 60
MAX_OUTPUT_QUEUE_SIZE = 60

logger = logging.getLogger("goggle_detection")

def load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_labels(labels_path) -> list:
    """Loads class labels, one per line."""
    path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _copy_inference_result(raw):
    """Copies buffers, since the bindings reuse them across frames."""
    if isinstance(raw, (list, tuple)):
        return [np.array(item) for item in raw]
    return np.array(raw)


def run_inference_pipeline(network: str,
                           input_queue: queue.Queue,
                           output_queue: queue.Queue,
                           stop_event: threading.Event) -> None:
    """
    Spins up a loop that performs inference on any preprocessed frames added to
    the input queue.

    Params:
        network: Neural network (HEF) to perform inference with.
        input_queue: Queue of (original_frame, preprocessed_frame) tuples.
        output_queue: Queue of (original_frame, raw_inference) tuples.
        stop_event: Event used to stop this pipeline.
    """
    params = VDevice.create_params()
    params.group_id = "SHARED"
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(network)
        infer_model.input().set_format_type(FormatType.UINT8)

        for name in infer_model.output_names:
            infer_model.output(name).set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured_infer_model:
            bindings = configured_infer_model.create_bindings()
            buffer_in = np.empty(infer_model.input().shape, dtype=np.uint8)
            bindings.input().set_buffer(buffer_in)
            for name in infer_model.output_names:
                out_info = infer_model.output(name)
                bindings.output(name).set_buffer(
                    np.empty(out_info.shape, dtype=np.float32)
                )

            while not stop_event.is_set():
                try:
                    frame = input_queue.get(timeout=0.5)  # blocks, times out to check stop_event
                except queue.Empty:
                    continue

                original_frame, preprocessed_frame = frame
                buffer_in[:] = preprocessed_frame
                configured_infer_model.run([bindings], timeout=TIMEOUT_MS)

                if len(infer_model.output_names) == 1:
                    buffer = bindings.output().get_buffer()
                    result = _copy_inference_result(buffer)
                else:
                    result = {
                        name: np.expand_dims(
                            _copy_inference_result(bindings.output(name).get_buffer()),
                            axis=0
                        )
                        for name in infer_model.output_names
                    }
                output_queue.put((original_frame, result))


def visualize(output_queue: queue.Queue,
              callback,
              stop_event: threading.Event,
              fps_tracker: FrameRateTracker | None = None,
              video_path: Path | None = None) -> None:
    """
    Applies the post-processing/visualization callback to each result and
    displays it. Frames are shown, and recorded, at their own resolution.

    If video_path is provided, the video writer is sized from the first
    annotated frame.
    """
    video_writer = None
    cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

    try:
        while not stop_event.is_set():
            try:
                result = output_queue.get(timeout=0.5)  # blocks, times out to check stop_event
            except queue.Empty:
                continue

            original_frame, inference_result = result
            frame_with_detections = callback(original_frame, inference_result)

            if fps_tracker is not None:
                fps_tracker.increment()

            if video_path is not None and video_writer is None:
                height, width = frame_with_detections.shape[:2]
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter.fourcc(*"XVID"),
                    CAMERA_SOURCE_FPS,
                    (width, height),
                )

            cv2.imshow("Output", frame_with_detections)

            if video_writer is not None:
                video_writer.write(frame_with_detections)
    finally:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Runs the camera, preprocessing, inference and visualization pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    network = args.hef_path or str(DEFAULT_HEF_PATH)
    labels = get_labels(args.labels or str(DEFAULT_LABELS_PATH))
    config_data = load_json_file(DEFAULT_CONFIG_PATH)

    video_path = None
    if args.save_output:
        video_path = Path(args.output_dir or DEFAULT_OUTPUT_DIR) / "output.avi"

    cap = init_rpicam2(CAMERA_WIDTH, CAMERA_HEIGHT)
    assert cap is not None, "Failed to open the Raspberry Pi camera."

    hef = HEF(network)
    model_input_height, model_input_width, _ = hef.get_input_vstream_infos()[0].shape
    assert model_input_width == model_input_height, "We only support models with square input widths/heights"

    stop_event = threading.Event()

    fps_tracker = FrameRateTracker()

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    visualizer_callback = partial(
        inference_result_handler,
        labels=labels,
        config_data=config_data,
    )

    inference_thread = threading.Thread(
        target=run_inference_pipeline,
        args=(network, input_queue, output_queue, stop_event),
    )
    visualize_thread = threading.Thread(
        target=visualize,
        args=(output_queue, visualizer_callback, stop_event, fps_tracker, video_path),
    )

    inference_thread.start()
    visualize_thread.start()
    fps_tracker.start()

    try:
        while True:
            ret, frame = cap.read()
            if ret:
                input_queue.put((frame, letterbox_img(frame, size=model_input_width, fill=114)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_event.set()
        inference_thread.join()
        visualize_thread.join()

        logger.info(fps_tracker.frame_rate_summary())
        if video_path is not None:
            logger.info(f"Results have been saved to {video_path}")


if __name__ == "__main__":
    main()
