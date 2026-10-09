from common.tools import init_rpicam2
from scrfd_detection import *
from visualizer import Visualizer, FrameSaver
import argparse
import threading

parser = argparse.ArgumentParser(description="Run SCRFD face detection with a live visualizer.")
parser.add_argument("--save-interval", type=float, default=None,
                    help="Save a frame every N seconds (0 = every frame). Omit to disable saving.")
parser.add_argument("--save-dir", default="captures", help="Directory to save frames into.")
parser.add_argument("--no-labels", action="store_true", help="Don't write YOLO label files next to images.")
parser.add_argument("--only-with-faces", action="store_true", help="Only save frames that contain a face.")
parser.add_argument("--crop-faces", action="store_true",
                    help="Save only the cropped face images instead of full frames (no label files).")
args = parser.parse_args()

camera = init_rpicam2(640, 480)

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
threads = [preprocess_thread, infer_thread, postprocess_thread]

for t in threads:
    t.start()

saver = None
if args.save_interval is not None:
    saver = FrameSaver(args.save_dir, args.save_interval,
                       save_labels=not args.no_labels, only_with_faces=args.only_with_faces, crop_faces=args.crop_faces)
visualizer = Visualizer(saver=saver)

try:
    while True:
        ok, frame = camera.read()
        if ok:
            input_queue.put(frame)

        # Drain the queue so we always display the most recent result
        latest = None
        while True:
            try:
                latest = outputs.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            key = visualizer.show(latest)
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
finally:
    stop_event.set()
    for t in threads:
        t.join(timeout=2)
    camera.release()
    visualizer.close()
