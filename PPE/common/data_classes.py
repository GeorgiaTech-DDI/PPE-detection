from dataclasses import dataclass
from typing import Optional

import cv2

@dataclass
class InputContext:
    """
    Context regarding input device for inference pipelines.
    """
    batch_size: int
    resolution: Optional[str]

    height: int
    width: int

    frame_rate: int

    cap: Optional[cv2.VideoCapture]

@dataclass
class VisualizationSettings:
    """
    Settings for visualization of inference output.
    """
    output_dir: str
    save_stream_output: bool
    output_resolution: tuple[int, int]