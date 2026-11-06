import sys
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

GOGGLE_LABELS_PATH = SCRIPT_DIR / "labels" / "goggles.txt"
FACE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "scrfd_500m.hef"
GOGGLE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "goggles_hybrid_dataset.hef"
SCORE_THRESHOLD = 0.4
MAX_BOXES_PER_IMAGE = 500

LED_WIDTH = 32
LED_HEIGHT = 8
LED_PIN = 18 # change if your data pin is different
LED_BRIGHTNESS = 80
LED_PIXEL_ORDER = "GRB"
LED_SIMULATE = False  # set True if testing without hardware
LED_FREQUENCY_HZ = 800000
LED_INVERT = False
LED_CHANNEL = 0
LED_DMA = 10