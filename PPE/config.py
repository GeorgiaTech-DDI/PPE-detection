import sys
from pathlib import Path
from argparse import ArgumentParser

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

GOGGLE_LABELS_PATH = SCRIPT_DIR / "labels" / "goggles.txt"
FACE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "scrfd_500m.hef"
GOGGLE_DETECT_HEF_PATH = SCRIPT_DIR / "models" / "hybrid_dataset_model.hef"
SCORE_THRESHOLD = 0.4
MAX_BOXES_PER_IMAGE = 500