import time
from pathlib import Path

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
# SCRFD landmark order: left eye, right eye, nose, left mouth corner, right mouth corner
LANDMARK_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, inferences: dict, draw_landmarks: bool = True) -> np.ndarray:
    """
    Draws SCRFD detections onto a copy of the frame. The original frame is left untouched.
    Params:
        frame: Image the inference was run on (640x640 for SCRFD).
        inferences: Output of SCRFDPostProc.tf_postproc. Boxes/landmarks are normalized to [0, 1].
        draw_landmarks: Whether to draw the 5 facial landmarks.
    Returns:
        Annotated copy of the frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    landmarks = inferences.get("face_landmarks")

    for i, (box, score) in enumerate(zip(inferences["detection_boxes"], inferences["detection_scores"])):
        x1, y1, x2, y2 = box
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(canvas, p1, p2, BOX_COLOR, 2)

        # Filled label background above the box (or inside it if the box touches the top edge)
        label = f"face {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
        label_y = p1[1] if p1[1] - th - baseline >= 0 else p1[1] + th + baseline
        cv2.rectangle(canvas, (p1[0], label_y - th - baseline), (p1[0] + tw, label_y), BOX_COLOR, -1)
        cv2.putText(canvas, label, (p1[0], label_y - baseline), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

        if draw_landmarks and landmarks is not None:
            for (lx, ly), color in zip(landmarks[i].reshape(-1, 2), LANDMARK_COLORS):
                cv2.circle(canvas, (int(lx * w), int(ly * h)), 3, color, -1)

    return canvas


class FrameSaver:
    """
    Periodically saves clean (un-annotated) frames to disk for building a dataset.
    Each image is saved alongside a YOLO-format label file: one "class cx cy w h" line per face, normalized to [0, 1].
    """

    def __init__(self, save_dir: str, interval_s: float, save_labels: bool = True, only_with_faces: bool = False):
        """
        Params:
            save_dir: Directory to write images (and labels) into. Created if it doesn't exist.
            interval_s: Minimum number of seconds between saved frames. 0 saves every frame.
            save_labels: Whether to write a .txt label file next to each image.
            only_with_faces: Skip frames where no face was detected.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self.save_labels = save_labels
        self.only_with_faces = only_with_faces
        self.num_saved = 0
        self._last_save = None

    def maybe_save(self, output: dict) -> bool:
        """
        Saves the frame in a postprocess output if the interval has elapsed.
        Returns:
            True if a frame was saved.
        """
        now = time.monotonic()
        if self._last_save is not None and now - self._last_save < self.interval_s:
            return False
        if self.only_with_faces and output["inferences"]["num_detections"] == 0:
            return False

        self._last_save = now
        stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.num_saved:06d}"
        cv2.imwrite(str(self.save_dir / f"{stem}.jpg"), output["frame"])

        if self.save_labels:
            lines = []
            for x1, y1, x2, y2 in np.clip(output["inferences"]["detection_boxes"], 0.0, 1.0):
                lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            (self.save_dir / f"{stem}.txt").write_text("\n".join(lines))

        self.num_saved += 1
        return True


class Visualizer:
    """
    Displays postprocessed SCRFD outputs in an OpenCV window with boxes, landmarks and an FPS counter.
    Optionally saves frames through a FrameSaver.
    """

    def __init__(self, window_name: str = "SCRFD Face Detection", draw_landmarks: bool = True,
                 saver: FrameSaver | None = None):
        self.window_name = window_name
        self.draw_landmarks = draw_landmarks
        self.saver = saver
        self._last_time = None
        self._fps = 0.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _update_fps(self):
        now = time.perf_counter()
        if self._last_time is not None:
            # Exponential moving average so the number doesn't jitter
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(now - self._last_time, 1e-6))
        self._last_time = now

    def render(self, output: dict) -> np.ndarray:
        """
        Builds the annotated frame for a postprocess output ({"frame", "inferences", "faces"}).
        """
        canvas = draw_detections(output["frame"], output["inferences"], self.draw_landmarks)
        status = f"FPS: {self._fps:.1f}  Faces: {output['inferences']['num_detections']}"
        if self.saver is not None:
            status += f"  Saved: {self.saver.num_saved}"
        cv2.putText(canvas, status, (10, 25), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def show(self, output: dict) -> int:
        """
        Renders and displays a postprocess output.
        Returns:
            Key code pressed during this frame (255 if none), so the caller can handle quit/save/etc.
        """
        self._update_fps()
        if self.saver is not None:
            self.saver.maybe_save(output)
        cv2.imshow(self.window_name, self.render(output))
        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)
