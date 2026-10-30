import queue
import threading
import supervision as sv
from dataclasses import dataclass
from typing import Optional

from hailo_platform import VDevice, FormatType, ConfigureParams, HailoSchedulingAlgorithm

from common.utils import letterbox_img
from scrfd_postproc import *

TIMEOUT_MS = 1000
CROP_BUFFER = 20

# Distance Estimation configuration. Height and width numbers are in normalized coordinates.
# Minimum of -1.5 standard deviation from mean allowed before faces are dropped.
MEAN_FACE_WIDTH_1M = 0.062864
MEAN_FACE_HEIGHT_1M = 0.077907
STDEV_FACE_WIDTH_1M = 0.007918
STDEV_FACE_HEIGHT_1M = 0.014038
MIN_Z_SCORE = -1.5
# ByteTrack configuration. The activation threshold is deliberately aligned with the
# SCRFD postprocessing score threshold so that every face we detect is eligible to
# start a track instead of being dropped by the tracker on its first frame.
TRACK_ACTIVATION_THRESHOLD = 0.3
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8
FRAME_RATE = 30
MINIMUM_CONSECUTIVE_FRAMES = 1

# Types
@dataclass
class PreprocessedFrame:
    hires_frame: np.ndarray
    lores_frame: np.ndarray

@dataclass
class SCRFDOutput:
    inferences: PostProcessedSCRFDInference
    hires_frame: np.ndarray
    lores_frame: np.ndarray
    faces: Optional[list[np.ndarray]] = None
    faces_by_id: Optional[dict[int, np.ndarray]] = None

def track_faces(postprocessed: SCRFDOutput,
                tracker: sv.ByteTrack,
                include: Optional[list[bool]] = None) -> SCRFDOutput:
    """
    Assigns persistent track IDs to the faces detected in a single frame.

    The SCRFD postprocessor reports boxes normalized to the preprocessed frame, so
    they are converted back to pixel coordinates before being handed to ByteTrack.
    IoU based association is only meaningful in unnormalized coordinates.

    ByteTrack only reports detections it managed to associate with a track, so
    faces_by_id contains only crops associated with a track. The faces field remains
    unchanged. A face missing from one frame may reappear under the same ID if the
    tracker can re-associate it.

    Params:
        postprocessed: SCRFDOutput containing the frame's detections, image frames,
            and face crops.
        tracker: Stateful sv.ByteTrack instance. Reuse it across frames from the
            same stream so track IDs remain consistent.
        include: Optional mask, one entry per detection. Only detections marked True
            are handed to ByteTrack, so excluded faces never get a track ID or appear
            in faces_by_id. None tracks every detection.

    Returns:
        The same SCRFDOutput instance, with faces_by_id set to a dictionary mapping
        each ByteTrack ID to its corresponding tracked face crop. The detections,
        frames, and faces fields are otherwise preserved.
    """
    inferences = postprocessed.inferences
    faces = postprocessed.faces

    img_h, img_w = postprocessed.lores_frame.shape[:2]
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

    postprocessed.faces_by_id = dict(zip(tracker_ids.tolist(), tracked_faces))

    return postprocessed

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
        output_queue: queue.Queue[SCRFDOutput],
        stop_event: threading.Event,
        track_only_within_1m: bool = False):
    """
    Spins up a loop that performs inference on any images added to the input queue.
    Params:
        network: Neural network to perform inference with.
        input_queue: Queue of camera frames, preprocessed.
        output_queue: Outputs from the neural network. Queue of SCRFDOutput
        stop_event: Event used to stop this pipeline.
        track_only_within_1m: If True, only faces estimated to be within 1m are given
            to ByteTrack, so faces_by_id only contains faces within 1m.
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

                # Inference
                buffer_in[:] = frame.lores_frame
                configured_infer_model.run([bindings], timeout=TIMEOUT_MS)

                # Run postprocessing
                inferences = postproc.postprocess(
                    [
                        bindings.output(name).get_buffer().copy()
                        for name in infer_model.output_names
                    ]
                )
                # Set up variables for postprocess cropping
                img_h, img_w = frame.hires_frame.shape[:2]
                crop_buffer = round(CROP_BUFFER * img_w / 640)
                cropped_faces = []
                inferences.within_1m = []
                for (x_min, y_min, x_max, y_max) in inferences.detection_boxes:
                    # Find normalized width and height, compare to mean
                    width = x_max - x_min
                    height = y_max - y_min
        
                    width_z_score = (width - MEAN_FACE_WIDTH_1M) / STDEV_FACE_WIDTH_1M
                    height_z_score = (height - MEAN_FACE_HEIGHT_1M) / STDEV_FACE_HEIGHT_1M
        
                    inferences.within_1m.append(True if width_z_score > MIN_Z_SCORE or height_z_score > MIN_Z_SCORE else False)
        
                    # Convert normalized coords to pixel coords
                    px_min = x_min * img_w
                    px_max = x_max * img_w
                    py_min = y_min * img_h
                    py_max = y_max * img_h
        
                    width = px_max - px_min
                    height = py_max - py_min
        
                    crop_x1 = max(0, int(px_min - crop_buffer))
                    crop_y1 = max(0, int(py_min - crop_buffer))
                    crop_x2 = min(img_w, int(px_max + crop_buffer))
                    crop_y2 = min(img_h, int(py_max + crop_buffer))
        
                    cropped_faces.append(frame.hires_frame[crop_y1:crop_y2, crop_x1:crop_x2])

                output = SCRFDOutput(
                    inferences=inferences,
                    hires_frame=frame.hires_frame,
                    lores_frame=frame.lores_frame,
                    faces=cropped_faces,
                )
                output = track_faces(
                    output,
                    bytetracker,
                    include=inferences.within_1m if track_only_within_1m else None,
                )

                output_queue.put(output)