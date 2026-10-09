import cv2
import numpy as np
import queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import threading
import supervision as sv

from hailo_platform import VDevice, FormatType, ConfigureParams, HailoSchedulingAlgorithm
from matplotlib.image import imread
from scrfd_postproc import *
from functools import partial

TIMEOUT_MS = 1000
CROP_BUFFER = 20

# ByteTrack configuration. The activation threshold is deliberately aligned with the
# SCRFD postprocessing score threshold so that every face we detect is eligible to
# start a track instead of being dropped by the tracker on its first frame.
TRACK_ACTIVATION_THRESHOLD = 0.3
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 30
MINIMUM_CONSECUTIVE_FRAMES = 1


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
    """
    Spins a thread which performs continuous preprocessing as frames come in.
    Params:
        input_queue: Queue of np.ndarray representing image frames in the format typically produced by OpenCV
        output_queue: Queue of preprocessed frames as np.ndarray. These are scaled to 640x640 for inference
        stop_event: threading.Event which stops this thread.
    """
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

    def callback(completion_info, bindings, result_queue, input_frame):
        if completion_info.exception:
            raise completion_info.exception
        outputs = {
            "frame": input_frame,
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
                bindings.output(name).set_buffer(np.empty(out_info.shape, dtype=np.float32))

            # Spin the thread (Wheeeeeeee!)
            while not stop_event.is_set():
                try:
                    frame = input_queue.get(timeout=0.5)   # blocks, times out to check stop_event
                except queue.Empty:
                    continue

                buffer_in[:] = frame
                configured_infer_model.run_async(
                    [bindings], 
                    partial(callback, bindings=bindings, result_queue=output_queue, input_frame=buffer_in.copy())
                )



def run_postprocess_pipeline(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
    """
    Postprocesses frames from SCRFD detection. Expects all postprocessing to be done on 640x640 frames.
    Params:
        input_queue: Queue containing dicts with "frame" mapped to the inference input, "raw_inference" to its output.
        output_queue: Queue for postprocessed frames and inferences with NMS applied.
            Contains labels "inferences" for postprocessed inferences and "faces" for cropped face images.
            Additionally contains label "frame" for the original frame that was given
        stop_event: threading.Event to stop this pipeline.
    """
    postproc = SCRFDPostProc((640,640))

    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        output = {}

        output["inferences"] = postproc.postprocess(input["raw_inference"])
        img_h, img_w = input["frame"].shape[0], input["frame"].shape[1]

        cropped_faces = []
        for (x_min, y_min, x_max, y_max) in output["inferences"]["detection_boxes"]:
            # Convert normalized coords to pixel coords
            px_min = x_min * img_w
            px_max = x_max * img_w
            py_min = y_min * img_h
            py_max = y_max * img_h

            width = px_max - px_min
            height = py_max - py_min

            crop_x1 = max(0, int(px_min - CROP_BUFFER))
            crop_y1 = max(0, int(py_min - CROP_BUFFER))
            crop_x2 = min(img_w, int(px_max + CROP_BUFFER))
            crop_y2 = min(img_h, int(py_max + CROP_BUFFER))

            cropped_faces.append(input["frame"][crop_y1:crop_y2, crop_x1:crop_x2])
        # Note that track IDs are assigned by run_tracking_pipeline, which sits
        # downstream of this stage. That way with multiple people within range, we
        # will still be able to tell whether someone was wearing glasses in the
        # last x time units.
        # Also we need to add logic that ignores faces too far away
        output["faces"] = cropped_faces
        output["frame"] = input["frame"]

        output_queue.put(output)


def track_faces(postprocessed: dict, tracker: sv.ByteTrack) -> dict:
    """
    Assigns persistent track IDs to the faces detected in a single frame.

    The SCRFD postprocessor reports boxes normalized to the preprocessed frame, so
    they are converted back to pixel coordinates before being handed to ByteTrack.
    IoU based association is only meaningful in unnormalized coordinates.

    ByteTrack only reports detections it managed to associate with a track, so
    "faces" is a subset of the crops produced by the postprocessing stage. A face
    missing from one frame simply reappears under the same ID once the tracker can
    re-associate it.

    Params:
        postprocessed: Dict produced by run_postprocess_pipeline, containing
            "inferences", "faces" and "frame".
        tracker: A stateful sv.ByteTrack instance. The same instance must be reused
            across frames for track IDs to be stable.

    Returns:
        Dict containing:
            "frame": The frame the detections were computed from.
            "detections": sv.Detections with a populated tracker_id field.
            "tracker_ids": np.ndarray holding the track ID for each entry in "faces".
            "faces": List of cropped face images aligned with "tracker_ids".
            "faces_by_track_id": Maps a track ID to its cropped face image.
    """
    inferences = postprocessed["inferences"]
    faces = postprocessed["faces"]
    frame = postprocessed["frame"]

    img_h, img_w = frame.shape[:2]
    xyxy = np.asarray(inferences["detection_boxes"], dtype=np.float32).reshape(-1, 4)
    xyxy = xyxy * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    scores = np.asarray(inferences["detection_scores"], dtype=np.float32).reshape(-1)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=scores,
        class_id=np.zeros(len(xyxy), dtype=int),
        data={"face_index": np.arange(len(xyxy))},
    )

    tracked = tracker.update_with_detections(detections)

    tracker_ids = tracked.tracker_id
    if tracker_ids is None:
        tracker_ids = np.array([], dtype=int)
    face_indices = np.asarray(tracked.data.get("face_index", []), dtype=int)
    tracked_faces = [faces[i] for i in face_indices if i < len(faces)]

    return {
        "inferences": postprocessed["inferences"],
        "frame": frame,
        "detections": tracked,
        "tracker_ids": tracker_ids,
        "faces": tracked_faces,
        "faces_by_track_id": dict(zip(tracker_ids.tolist(), tracked_faces)),
    }


def run_tracking_pipeline(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event):
    """
    Spins a thread that assigns persistent IDs to detected faces using ByteTrack.

    This is kept as a separate stage from postprocessing so that the tracker, which
    is stateful and not thread safe, is only ever touched by a single thread, and so
    that the frame order it sees is defined by this queue alone.

    Params:
        input_queue: Queue containing dicts produced by run_postprocess_pipeline.
        output_queue: Queue for the tracked results. Each entry is the input dict
            augmented with the keys added by track_faces, most notably
            "tracker_ids" and "faces_by_track_id".
        stop_event: threading.Event to stop this pipeline.
    """
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
        frame_rate=FRAME_RATE,
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
    )

    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        result = track_faces(input, tracker)
        output_queue.put(result)

        if len(result["faces_by_track_id"]) == 0:
            no_faces_count += 1
        else:
            no_faces_count = 0
        if no_faces_count >= 20:
            tracker.reset()
            print("Reset Tracker")
            no_faces_count = 0
