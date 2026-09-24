import cv2
import numpy as np
import queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import threading

from hailo_platform import VDevice, FormatType, ConfigureParams, HailoSchedulingAlgorithm
from matplotlib.image import imread
from scrfd_postproc import *
from functools import partial

timeout_ms = 1000


# Takes whatever size input, scales to 640x640 with letterboxes.
# Returns Numpy ndarray
def preprocess(image: np.ndarray):
    h, w = image.shape[:2]

    scale = min(640 / w, 640 / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    image = cv2.resize(image, (new_w, new_h))

    pad_x = 640 - new_w
    pad_y = 640 - new_h

    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return np.ascontiguousarray(image)

def run_preprocess_pipeline(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            input = input_queue.get()
        except queue.Empty:
            continue

        output_queue.put(preprocess(input))



def run_inference_pipeline(
        network: str,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event):
    """
    Spins up a loop that performs inference on any preprocessed images added to the input queue.
    Params:
        network: Neural network to perform inference with.
        input_queue: The inputs to the neural network. For SCRFD models, expects a 640x640 image.
        output_queue: Outputs from the neural network. Queue of dicts containing copy of input and its output.
        stop_event: Event used to stop this pipeline. 
    """

    def callback(completion_info, bindings, result_queue):
        if completion_info.exception:
            raise completion_info.exception
        outputs = {
            "input": bindings.input(),
            "raw_inference": [bindings.output(name).get_buffer().copy()
            for name in infer_model.output_names]
        }
        result_queue.put(outputs)
    
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
                bindings.output(name).set_buffer(np.empty(out_info.shape, dtype=out_info.dtype))

            # Spin the thread (Wheeeeeeee!)
            while not stop_event.is_set():
                try:
                    frame = input_queue.get(timeout=0.5)   # blocks, times out to check stop_event
                except queue.Empty:
                    continue

                buffer_in[:] = frame
                configured_infer_model.run_async(
                    [bindings], 
                    partial(callback, bindings=bindings, result_queue=output_queue)
                )



def run_postprocess_pipeline(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
    """
    Postprocesses frames from SCRFD detection. Expects all postprocessing to be done on 640x640 frames.
    Params:
        input_queue: Queue containing dicts with "input" mapped to the inference input, "raw_inference" to its output.
        output_queue: Queue for postprocessed frames and inferences with NMS applied.
            Contains labels "inferences" for postprocessed inferences and "faces" for cropped face images.
        stop_event: threading.Event to stop this pipeline.
    """
    postproc = SCRFDPostProc((640,640))

    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        output = {}

        output['inferences'] = postproc.tf_postproc([tf.convert_to_tensor(i) for i in input["raw_inference"]])
        img_h, img_w = input["input"].shape[0], input["input"].shape[1]

        buffer = 20
        cropped_faces = []
        for (x_min, y_min, x_max, y_max) in output['inferences']["detection_boxes"]:
            # Convert normalized coords to pixel coords
            px_min = x_min * img_w
            px_max = x_max * img_w
            py_min = y_min * img_h
            py_max = y_max * img_h

            width = px_max - px_min
            height = py_max - py_min

            crop_x1 = max(0, int(px_min - buffer))
            crop_y1 = max(0, int(py_min - buffer))
            crop_x2 = min(img_w, int(px_max + buffer))
            crop_y2 = min(img_h, int(py_max + buffer))

            cropped_faces.append(input['input'][crop_y1:crop_y2, crop_x1:crop_x2])
        # Note that we may want to apply ByteTrack here so we can identify which face belongs to who
        # That way with multiple people within range, we will still be able to detect whether someone
        # was wearing glasses in the last x time units.
        # Also we need to add logic that ignores faces too far away
        output["faces"] = cropped_faces

        output_queue.put(output)






# The vdevice is used as a context manager ("with" statement) to ensure it's released on time.
# with VDevice() as vdevice:
#     print("VDevice created successfully")

#     # Create an infer model from an HEF:
#     infer_model = vdevice.create_infer_model('scrfd_10g.hef')
#     print(f"Model loaded: input shape {infer_model.input().shape}")
#     print("Output shapes are:")
#     for i in infer_model.output_names:
#         print(f"{i}: {infer_model.output(i).shape}")

#     img = preprocess("hide_the_pain_harold.jpg")

#     for name in infer_model.output_names:
#         infer_model.output(name).set_format_type(FormatType.FLOAT32)

#     # Configure the infer model and create bindings for it
#     with infer_model.configure() as configured_infer_model:
#         print("Model configured")
#         bindings = configured_infer_model.create_bindings()

#         # Set input and output buffers
#         buffer = np.zeros(infer_model.input().shape, dtype=np.uint8)
#         bindings.input().set_buffer(img)

#         for i in infer_model.output_names:
#             buffer = np.zeros(infer_model.output(i).shape, dtype=np.float32)
#             bindings.output(i).set_buffer(buffer)

#         # Run synchronous inference and access the output buffers
#         print("Running synchronous inference...")
#         configured_infer_model.run([bindings], timeout_ms)
#         outputs = []
#         for i in infer_model.output_names:
#             buffer = bindings.output(i).get_buffer()
#             outputs.append(bindings.output(i).get_buffer())
#             print(f"Synchronous inference done - output shape: {buffer.shape}")

#         postproc = SCRFDPostProc((640,640))

#         outputs = postproc.tf_postproc([tf.convert_to_tensor(i) for i  in outputs])
#         print(outputs)

#         img = imread("./hide_the_pain_harold.jpg")
#         img_h, img_w = img.shape[0], img.shape[1]
#         fig, ax = plt.subplots(1)
#         ax.imshow(img)

#         for (x_min, y_min, x_max, y_max) in outputs["detection_boxes"]:
#             # Convert normalized coords to pixel coords
#             px_min = x_min * img_w
#             px_max = x_max * img_w
#             py_min = y_min * img_h
#             py_max = y_max * img_h

#             width = px_max - px_min
#             height = py_max - py_min

#             rect = patches.Rectangle(
#                 (px_min, py_min), width, height,
#                 linewidth=2, edgecolor="red", facecolor="none"
#             )
#             ax.add_patch(rect)

#         ax.axis("off")
#         plt.tight_layout()
#         plt.show()






