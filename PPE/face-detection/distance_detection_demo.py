from common.tools import init_rpicam2
from scrfd_detection import *
import threading
import argparse

import pandas as pd

"""
Demonstrates distance detection mechanism.
Distance detection 
"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "--save", 
    action="store_true", 
    help="Record face sizes to a specified distance"
)
parser.add_argument(
    "--dist",
    type=str,
    help="Distance from camera to subject for testing"
)
args = parser.parse_args()

SAVE_DATA = args.save
dist = args.dist

if SAVE_DATA and not dist:
    dist = input("What distance is this data for? ")

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
        inferences = output.get('inferences')
        if frames and inferences:
            if SAVE_DATA:
                det = inferences["detection_boxes"][0]
                rows.append({
                    "width": det[2] - det[0],
                    "height": det[3] - det[1],
                })
                cv2.imshow("First Face", frames[0])
            else:
                if (inferences["within_1m"][0] == False):
                    continue
                det = inferences["detection_boxes"][0]
                cv2.imshow("First Face", frames[0])
    except queue.Empty:
        pass
    if cv2.waitKey(1) == ord('q'):
        break

stop_event.set()

if SAVE_DATA:
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"face_sizes_{dist}.csv", index=False)
        with open(f"face_size_stats_{dist}.txt", "w") as f:
            f.write(f"Distance: {dist}\n")
            f.write(f"Samples: {len(df)}\n")
            f.write(df.agg(["mean", "std"]).to_string())
    else:
        print("No faces captured, nothing saved.")

exit()