import queue

import numpy as np


class HailoBindingsQueue:
    """
    Queue of available bindings for a given ConfiguredInferModel.
    When running asynchronous inference with batching, this must be used.
    Keeps an available set of bindings for a Hailo device.
    Each time an image is queued up to send to the Hailo device for processing,
    a bindings object should be taken from this object.
    If out of bindings, the caller will be blocked until a binding becomes available.
    Bindings objects should be returned after inference and postprocessing is complete.
    """

    def __init__(
        self,
        infer_model,
        configured_infer_model,
        size: int,
    ):
        if size <= 0:
            raise ValueError("size must be greater than zero")

        self._available = queue.Queue(maxsize=size)
        self._buffers = []

        for _ in range(size):
            bindings = configured_infer_model.create_bindings()
            input_buffer = np.empty(infer_model.input().shape, dtype=np.uint8)
            bindings.input().set_buffer(input_buffer)

            output_buffers = {}
            for name in infer_model.output_names:
                output_buffer = np.empty(
                    infer_model.output(name).shape,
                    dtype=np.float32,
                )
                bindings.output(name).set_buffer(output_buffer)
                output_buffers[name] = output_buffer

            self._buffers.append((input_buffer, output_buffers))
            self._available.put(bindings)

    def put(self, bindings) -> None:
        self._available.put(bindings)

    def get(self):
        return self._available.get()