from functools import partial
import queue
import threading
import supervision as sv
from dataclasses import dataclass
from typing import Optional
import config

from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

from camera_utils import letterbox_img
from hailo_bindings_queue import HailoBindingsQueue
from .scrfd_postproc import *

# ByteTrack configuration. The activation threshold is deliberately aligned with the
# SCRFD postprocessing score threshold so that every face we detect is eligible to
# start a track instead of being dropped by the tracker on its first frame.
TRACK_ACTIVATION_THRESHOLD = 0.3
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 30
MINIMUM_CONSECUTIVE_FRAMES = 1

# Types
Face = np.ndarray
ScrfdOutput = dict[int, Face]

@dataclass
class PreprocessedFrame:
    hires_frame: np.ndarray
    lores_frame: np.ndarray

def track_faces(inferences: PostProcessedSCRFDInference,
                faces: Optional[list[np.ndarray]],
                lores_frame: np.ndarray,
                tracker: sv.ByteTrack,
                include: Optional[list[bool]] = None) -> ScrfdOutput:
    """
    Assigns persistent track IDs to the faces detected in a single frame.

    The SCRFD postprocessor reports boxes normalized to the preprocessed frame, so
    they are converted back to pixel coordinates before being handed to ByteTrack.
    IoU based association is only meaningful in unnormalized coordinates.

    ByteTrack only reports detections it managed to associate with a track, so the
    result contains only crops associated with a track. A face missing from one frame
    may reappear under the same ID if the tracker can re-associate it.

    Params:
        inferences: The frame's SCRFD detections.
        faces: Face crops, one per detection, in detection order.
        lores_frame: Preprocessed frame the detection boxes are normalized to.
        tracker: Stateful sv.ByteTrack instance. Reuse it across frames from the
            same stream so track IDs remain consistent.
        include: Optional mask, one entry per detection. Only detections marked True
            are handed to ByteTrack, so excluded faces never get a track ID or appear
            in the result. None tracks every detection.

    Returns:
        Each ByteTrack ID mapped to its corresponding tracked face crop.
    """
    img_h, img_w = lores_frame.shape[:2]
    xyxy = np.asarray(inferences.detection_boxes, dtype=np.float32).reshape(-1, 4)
    xyxy = xyxy * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    scores = np.asarray(inferences.detection_scores, dtype=np.float32).reshape(-1)

    # Keep the original indices so tracked detections still map back to their crops.
    kept_indices = np.arange(len(xyxy))
    if include is not None:
        kept_indices = kept_indices[np.asarray(include, dtype=bool)]

    detections = sv.Detections(
        xyxy=xyxy[kept_indices],
        confidence=scores[kept_indices],
        class_id=np.zeros(len(kept_indices), dtype=int),
        data={"face_index": kept_indices},
    )

    tracked = tracker.update_with_detections(detections)

    tracker_ids = tracked.tracker_id
    if tracker_ids is None:
        tracker_ids = np.array([], dtype=int)
    face_indices = np.asarray(tracked.data.get("face_index", []), dtype=int)
    tracked_faces = [faces[i] for i in face_indices if i < len(faces)]

    return dict(zip(tracker_ids.tolist(), tracked_faces))


def run_preprocess_pipeline(
    input_queue: queue.Queue[np.ndarray],
    output_queue: queue.Queue[PreprocessedFrame],
    stop_event: threading.Event):
    """
    Spins up a loop that preprocesses any images so that they may have inference run on them.
    Params:
        input_queue: queue of camera frames
        output_queue: queue to which it should add its output, in format PreprocessedFrame
        stop_event: threading event to stop this pipeline.
    """
    while not stop_event.is_set():
        try:
            input = input_queue.get(timeout=0.5) #for checking stop event
        except queue.Empty:
            continue

        output_queue.put(
            PreprocessedFrame(
                lores_frame=letterbox_img(input, resize_interpolation=cv2.INTER_LINEAR),
                hires_frame=letterbox_img(input, size=1920, resize_interpolation=cv2.INTER_LINEAR),
            )
        )

def run_inference_pipeline(
        network: str,
        input_queue: queue.Queue[PreprocessedFrame],
        output_queue: queue.Queue[dict[int, np.ndarray]],
        stop_event: threading.Event):
    """
    Spins up a loop that performs inference on any images added to the input queue.
    Params:
        network: Neural network to perform inference with.
        input_queue: Queue of camera frames, preprocessed.
        output_queue: Tracked face crops, keyed by ByteTrack ID, one dict per frame.
        stop_event: Event used to stop this pipeline.
    """

    # Setup VDevice to be shared across frames with Round Robin scheuler
    params = VDevice.create_params()
    params.group_id = "SHARED"
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    postproc = SCRFDPostProc((640, 640))
    
    bytetracker = sv.ByteTrack(
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
        frame_rate=FRAME_RATE,
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
    )

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
                config.BATCH_SIZE
            )

            # Spin the thread (Wheeeeeeee!)
            while not stop_event.is_set():
                try:
                    frame = input_queue.get(timeout=0.5)   # blocks, times out to check stop_event
                except queue.Empty:
                    continue

                # Inference
                _bindings = available_bindings.get()
                _bindings.input().get_buffer()[:] = frame.lores_frame
                configured_infer_model.run_async(
                    [_bindings],
                    partial(scrfd_callback, 
                            frame=frame, 
                            bindings=_bindings, 
                            available_bindings_queue=available_bindings,
                            infer_model=infer_model, 
                            tracker=bytetracker,
                            output_queue=output_queue))
        

def scrfd_callback(
        completion_info, 
        bindings, 
        available_bindings_queue,
        frame, 
        infer_model, 
        tracker: sv.ByteTrack, 
        output_queue):
    """
    Callback for SCRFD asynchronous inference.
    Postprocesses, crops and tracks faces from SCRFD inference.
    
    """
    if completion_info.exception:
        exit(1)
    postproc = SCRFDPostProc((640, 640))

    endnodes = [
                    bindings.output(name).get_buffer().copy()
                    for name in infer_model.output_names
                ]
    inferences = postproc.postprocess(endnodes)
    cropped_faces = postproc.crop_faces(inferences, frame.hires_frame)
    output = SCRFDOutput(
        inferences=inferences,
        hires_frame=frame.hires_frame,
        lores_frame=frame.lores_frame,
        faces=cropped_faces,
    )
    output = track_faces(
        output,
        tracker,
        None
    )
    output_queue.put(output)
    available_bindings_queue.put(bindings)
