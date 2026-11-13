import sys
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Hailo Inference Parameters
QUEUE_SIZE = 50
LOOP_PERIOD_SECS = 0.01
BONUS_DELAY_PER_FACE_SECS = 0.01
FULL_QUEUE_FRAMES_TO_DROP = 10
NO_FACES_FRAME_THRESHOLD = 20
NO_FACES_DELAY = 0.2
BATCH_SIZE = 10

CROP_BUFFER = 20
TIMEOUT_MS = 1000

GOGGLE_LABELS_PATH = SCRIPT_DIR / "labels" / "goggles.txt"
FACE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "scrfd_500m.hef"
GOGGLE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "goggles_hybrid_dataset.hef"
SCORE_THRESHOLD = 0.4
MAX_BOXES_PER_IMAGE = 500
YOLO_MODEL_VERSION = 11

LED_BRIGHTNESS = 80
LED_PIXEL_ORDER = "GRB"
LED_SIMULATE = False  # set True if testing without hardware
LED_PIXEL_COUNT = 8