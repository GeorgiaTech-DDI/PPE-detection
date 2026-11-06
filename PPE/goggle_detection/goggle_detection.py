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

                original_frame, preprocessed_frame = frame
                _bindings = available_bindings.get()
                _bindings.input().get_buffer()[:] = preprocessed_frame
                configured_infer_model.run_async([_bindings], partial(
                    goggle_pipeline_callback,
                    bindings=_bindings,
                    bindings_queue=available_bindings,
                    original_frame=original_frame,
                    infer_model=infer_model,
                    output_queue=output_queue
                ))

                

def goggle_pipeline_callback(
        completion_info, 
        bindings, 
        bindings_queue,
        original_frame, 
        infer_model, 
        output_queue):
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
    bindings_queue.put(bindings)
