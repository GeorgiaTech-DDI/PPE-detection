from common.tools import init_rpicam2
from scrfd_detection import *
import threading

import pandas as pd

camera = init_rpicam2(1920, 1080)

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

rows = []

while True:
    input_queue.put(camera.read()[1])
    try:
        output = outputs.get_nowait()
        frames = output.get('faces')
        dets = output.get('inferences')
        if frames and dets:
            det = dets[0]
            rows.append({
                "width": det[2] - det[0],
                "height": det[3] - det[1],
            })
            cv2.imshow("First Face", frames[0])
    except queue.Empty:
        pass
    if cv2.waitKey(1) == ord('q'):
        break

stop_event.set()

if rows:
    df = pd.DataFrame(rows)
    summary = df.agg(["mean", "std"])
    pd.concat([df, summary]).to_csv("face_sizes.csv", index_label="row")
    print(summary)