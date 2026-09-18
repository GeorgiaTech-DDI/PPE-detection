#!/usr/bin/env python3
import cv2
import os
import sys
import queue
import threading
from functools import partial
from types import SimpleNamespace
import numpy as np
from pathlib import Path
import collections
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from tools import init_rpicam2
# TODO: Remove dependency on hailo-apps as it is a large, unweildy, and possibly unreliable library
# HailoRT's python API is much more barebones but is also more well-maintained.
sys.path.append("/home/pi/hailo-apps")
try:
    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.camera_utils import select_cap_processing_mode
    from hailo_apps.python.core.common.toolbox import (
        InputContext,
        VisualizationSettings,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        FrameRateTracker,
        InputType,
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import inference_result_handler
    from hailo_apps.python.core.common.core import handle_and_resolve_args
except ImportError:
    # Running as a plain script: add repo root so `import hailo_apps` works.
    repo_root = None
    for p in Path(__file__).resolve().parents:
        if (p / "hailo_apps" / "config" / "config_manager.py").exists():
            repo_root = p
            break
    if repo_root is not None:
        sys.path.insert(0, str(repo_root))

    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker
    from hailo_apps.python.core.common.hailo_inference import HailoInfer
    from hailo_apps.python.core.common.core import handle_and_resolve_args
    from hailo_apps.python.core.common.toolbox import (
        init_input_source,
        get_labels,
        load_json_file,
        preprocess,
        visualize,
        select_cap_processing_mode,
        FrameRateTracker,
    )
    from hailo_apps.python.core.common.defines import (
        MAX_INPUT_QUEUE_SIZE,
        MAX_OUTPUT_QUEUE_SIZE,
        MAX_ASYNC_INFER_JOBS
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.hailo_logger import get_logger, init_logging, level_from_args
    from goggle_post_process import inference_result_handler

APP_NAME = Path(__file__).stem
logger = get_logger(__name__)


def parse_args():
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Run object detection with optional tracking and performance measurement."

    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Enable object tracking for detections. "
            "When enabled, detected objects will be tracked across frames using a tracking algorithm "
            "(e.g., ByteTrack). This assigns consistent IDs to objects over time, enabling temporal analysis, "
            "trajectory visualization, and multi-frame association. Useful for video processing applications."
        ),
    )

    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        ),
    )

    parser.add_argument(
        "--draw-trail",
        action="store_true",
        help=(
            "[Tracking only] Draw motion trails of tracked objects.\n"
            "Uses the last 30 positions from the tracker history."
        )
    )

    args = parser.parse_args()
    return args


def run_inference_pipeline(net, labels, input_context: InputContext,
                           visualization_settings: VisualizationSettings,
          enable_tracking=False, show_fps=False, draw_trail=False) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    labels = get_labels(labels)
    config_data = load_json_file("config.json")

    # Initialize rpicam2 using copy-pasted hailo-apps code
    # TODO: Uses a lot of copy-pasted code from hailo-apps,
    # maybe replace with something less dodgy
    input_context.cap = init_rpicam2()
    input_context.input_type = InputType.RPI_CAMERA
    input_context.source_fps = 30
    if input_context.cap is not None:
        input_context.width = int(input_context.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        input_context.height = int(input_context.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    input_context.cap_processing_mode = select_cap_processing_mode(
        input_type=input_context.input_type.value,
        frame_rate=input_context.frame_rate,
        source_fps=input_context.source_fps,
        video_unpaced=input_context.video_unpaced,
    )

    stop_event = threading.Event()
    tracker = None
    fps_tracker = None
    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking:
        # load tracker config from config_data
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue(MAX_INPUT_QUEUE_SIZE)
    output_queue = queue.Queue(MAX_OUTPUT_QUEUE_SIZE)

    post_process_callback_fn = partial(
        inference_result_handler, labels=labels,
        config_data=config_data, tracker=tracker, draw_trail=draw_trail
    )

    hailo_inference = HailoInfer(net, input_context.batch_size)
    input_context.height, input_context.width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, args=(input_context, input_queue, input_context.width, input_context.height, None, stop_event)
    )
    postprocess_thread = threading.Thread(
        target=visualize, 
        args=(input_context, visualization_settings, output_queue, post_process_callback_fn, fps_tracker, stop_event)
    )
    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue, stop_event)
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    postprocess_thread.join()

    if show_fps:
        logger.info(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")
    if visualization_settings.save_stream_output or input_context.input_src.lower() not in ("usb", "rpi"):
        logger.info(f"Results have been saved in {visualization_settings.output_dir}")


def infer(hailo_inference, input_queue, output_queue, stop_event):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    # Limit number of concurrent async inferences
    pending_jobs = collections.deque()

    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        if stop_event.is_set():
            continue  # Skip processing if stop signal is set

        input_batch, preprocessed_batch = next_batch

        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )


        while len(pending_jobs) >= MAX_ASYNC_INFER_JOBS:
            pending_jobs.popleft().wait(10000)

        # Run async inference
        job = hailo_inference.run(preprocessed_batch, inference_callback_fn)
        pending_jobs.append(job)

    # Release resources and context
    hailo_inference.close()
    output_queue.put(None)


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))

def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    init_logging(level=level_from_args(args))
    handle_and_resolve_args(args, APP_NAME)
    input_context = InputContext(
        input_src=args.input,
        batch_size=args.batch_size,
        resolution=args.camera_resolution,
        frame_rate=args.frame_rate,
    )
    visualization_settings = VisualizationSettings(
        output_dir=args.output_dir,
        save_stream_output=args.save_output,
        output_resolution=args.output_resolution,
    )
    run_inference_pipeline(
        args.hef_path,
        args.labels,
        input_context,
        visualization_settings,
        args.track,
        args.show_fps,
        args.draw_trail
    )


if __name__ == "__main__":
    main()
