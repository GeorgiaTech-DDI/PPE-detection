import logging
import queue
import threading
import numpy as np
from functools import partial
from hailo_platform import VDevice, FormatType
from .data_classes import InputContext

logger = logging.getLogger("google_detection_logger")

def infer(
        device: VDevice,
        network: str,
        input_context: InputContext,
        input_queue: queue.Queue[np.ndarray], 
        output_queue: queue.Queue[np.ndarray], 
        stop_event: threading.Event,):

    infer_model = VDevice.create_infer_model(network)
    infer_model.set_batch_size(input_context.batch_size)

    infer_model.input().set_format_type(FormatType.FLOAT32)
    infer_model.output().set_format_type(FormatType.FLOAT32)

    with infer_model.configure() as configured_model:
        while True:
            next_batch = input_queue.get()
            if not next_batch:
                break
            if stop_event.is_set():
                continue

            input_batch, preprocessed_batch = next_batch
            output = np.empty(infer_model.output().shape).astype(np.float32)
            bindings = configured_model.create_bindings()
            bindings.input().set_buffer(preprocessed_batch).astype(np.float32)
            bindings.output().set_buffer(output)

            inference_callback_fn = partial(
                        inference_callback,
                        input_batch=input_batch,
                        output_queue=output_queue
                    )

            configured_model.wait_for_async_ready(timeout_ms=10000)

            job = configured_model.run_async([bindings], partial(inference_callback_fn, bindings=bindings))
        job.wait(10000)

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