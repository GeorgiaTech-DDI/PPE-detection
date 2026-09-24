from common.tools import init_rpicam2
from scrfd_detection import *
import threading

camera = init_rpicam2()

stop_event = threading.Event()

input_queue = queue.Queue()
preprocessed_frames = queue.Queue()
inferences = queue.Queue()
outputs = queue.Queue()


from pathlib import Path
hef_path = str(Path(__file__).resolve().parent / "scrfd_10g.hef")
preprocess_thread = threading.Thread(target=run_preprocess_pipeline, args=(input_queue, preprocessed_frames, stop_event))
infer_thread = threading.Thread(target=run_inference_pipeline, args=(hef_path,preprocessed_frames, inferences, stop_event))
postprocess_thread = threading.Thread(target=run_postprocess_pipeline, args=(inferences, outputs, stop_event))


preprocess_thread.start()
infer_thread.start()
postprocess_thread.start()

while True:
    input_queue.put(camera.read())

    try:
        output = outputs.get_nowait()
        frames = output.get('faces')
        if frames:
            frame = frames[0]
            cv2.imshow("First Face", frame)
    except queue.Empty:
        pass
    if cv2.waitKey(1) == ord('q'):
        break