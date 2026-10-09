from common.tools import init_rpicam2
from scrfd_detection import *
import threading

camera = init_rpicam2(1920, 1080)

stop_event = threading.Event()

input_queue = queue.Queue()
preprocessed_frames = queue.Queue()
inferences = queue.Queue()
outputs = queue.Queue()
tracked_outputs = queue.Queue()


from pathlib import Path
hef_path = str(Path(__file__).resolve().parent / "scrfd_10g.hef")
preprocess_thread = threading.Thread(target=run_preprocess_pipeline, args=(input_queue, preprocessed_frames, stop_event))
infer_thread = threading.Thread(target=run_inference_pipeline, args=(hef_path,preprocessed_frames, inferences, stop_event))
postprocess_thread = threading.Thread(target=run_postprocess_pipeline, args=(inferences, outputs, stop_event))
tracking_thread = threading.Thread(target=run_tracking_pipeline, args=(outputs, tracked_outputs, stop_event))
windows_cleared = False

preprocess_thread.start()
infer_thread.start()
postprocess_thread.start()
tracking_thread.start()

while True:
    frame = camera.read()[1]
    input_queue.put(frame)
    # frm = input_queue.get()
    # # print(frm)
    # cv2.imshow("yeet",frm)
    # if cv2.waitKey(1) == ord('q'):
    #     break

    try:
        faces = tracked_outputs.get_nowait()["faces_by_track_id"]
        if len(faces) == 0 and not windows_cleared:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            windows_cleared = True
            print("No Faces detected: stale windows closed")
        elif len(faces) != 0:
            windows_cleared = False
        # Faces reappear under the same key for as long as ByteTrack keeps the
        # track alive, so this is where "was person N wearing glasses" state
        # would be kept.
        for track_id, face in faces.items():
            cv2.imshow(f"Face {track_id}", face)
    except queue.Empty:
        pass
    if cv2.waitKey(1) == ord('q'):
        break