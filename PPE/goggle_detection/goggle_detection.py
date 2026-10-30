#!/usr/bin/env python3
import json
import logging
import queue
import sys
import threading
from pathlib import Path

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

DEFAULT_LABELS_PATH = SCRIPT_DIR / "goggle_labels.txt"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
TIMEOUT_MS = 10000

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
