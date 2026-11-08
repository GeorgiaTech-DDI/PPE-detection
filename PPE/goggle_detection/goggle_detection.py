#!/usr/bin/env python3
from functools import partial
import config
import json
import logging
import queue
import threading
from pathlib import Path
from hailo_bindings_queue import HailoBindingsQueue

import numpy as np
from hailo_platform import (
    FormatType,
    HEF,
    HailoSchedulingAlgorithm,
    VDevice,
)

TIMEOUT_MS = 10000

logger = logging.getLogger("goggle_detection")

def load_json_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_inference_result(raw) -> np.ndarray:
    """Copies buffers, since the bindings reuse them across frames.

    NMS outputs arrive as one array per class and the classes hold different
    numbers of detections, so the per-class arrays are padded up to a common
    length before stacking them into a dense [class, detection, 5] array. The
    padding is zeros, so its score is below any score threshold.
    """
    if not isinstance(raw, (list, tuple)):
        return np.array(raw)

    arrays = [np.asarray(item) for item in raw]
    if not arrays:
        return np.array(raw, dtype=np.float32)
    if len({array.shape for array in arrays}) == 1:
        return np.stack(arrays)

    max_detections = max(array.shape[0] for array in arrays)
    return np.stack([
        np.pad(array, ((0, max_detections - array.shape[0]), *((0, 0),) * (array.ndim - 1)))
        for array in arrays
    ])

LetterboxedImg = np.ndarray
GoggleInput = tuple[int, LetterboxedImg]
RawInference = dict[str, np.ndarray]
GoggleOutput = tuple[int, RawInference]

def run_inference_pipeline(network: str,
                           input_queue: queue.Queue[GoggleInput],
                           output_queue: queue.Queue[GoggleOutput],
                           stop_event: threading.Event) -> None:
    """
    Spins up a loop that performs inference on any preprocessed frames added to
    the input queue.

    Params:
        network: Neural network (HEF) to perform inference with.
        input_queue: Queue of (track_id, preprocessed_frame) tuples.
        output_queue: Queue of (track_id, raw_inference) tuples. The caller needs
            the track ID back to match each result to the face it came from.
        stop_event: Event used to stop this pipeline.
    """
    params = VDevice.create_params()
    params.group_id = "SHARED"
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(network)
        infer_model.set_batch_size(config.BATCH_SIZE)
        infer_model.input().set_format_type(FormatType.UINT8)

        for name in infer_model.output_names:
            infer_model.output(name).set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured_infer_model:

            available_bindings = HailoBindingsQueue(
                infer_model,
                configured_infer_model,
                config.BATCH_SIZE,
            )

            while not stop_event.is_set():
                try:
                    frame = input_queue.get(timeout=0.5)  # blocks, times out to check stop_event
                except queue.Empty:
                    continue

                track_id, preprocessed_frame = frame
                _bindings = available_bindings.get()
                _bindings.input().get_buffer()[:] = preprocessed_frame
                configured_infer_model.run_async([_bindings], partial(
                    goggle_pipeline_callback,
                    bindings=_bindings,
                    bindings_queue=available_bindings,
                    track_id=track_id,
                    infer_model=infer_model,
                    output_queue=output_queue
                ))

                

def goggle_pipeline_callback(
        completion_info, 
        bindings, 
        bindings_queue,
        track_id, 
        infer_model, 
        output_queue):
    result = {
        name: np.expand_dims(
            _copy_inference_result(bindings.output(name).get_buffer()),
            axis=0
        )
        for name in infer_model.output_names
    }
    output_queue.put((track_id, result))
    bindings_queue.put(bindings)
