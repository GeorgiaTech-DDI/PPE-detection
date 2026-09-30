from .data_classes import InputContext
import cv2
import threading

class PiCamera2CaptureAdapter:
    """
    Adapter that makes Picamera2 behave like cv2.VideoCapture.
    Copy of hailo-apps.python.common.camera_utils.PiCamera2CaptureAdapter
    Threading fixed to work on our system.
    """

    def __init__(self, picam2):
        self.picam2 = picam2
        self._opened = True
        self._io_lock = threading.Lock()

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None

        # prevent stop/close while capturing
        with self._io_lock:
            if not self._opened: # re-check after taking lock
                return False, None
            frame = self.picam2.capture_array()

        if frame is None:
            return False, None
        return True, frame

    def get(self, prop_id: int) -> float:
        if prop_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            try:
                cfg = self.picam2.camera_configuration()
                size = cfg.get("main", {}).get("size", None)
                if size and len(size) == 2:
                    w, h = int(size[0]), int(size[1])
                    return float(w if prop_id == cv2.CAP_PROP_FRAME_WIDTH else h)
            except Exception:
                pass
            return 0.0
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        return None

    def release(self):
        # stop new reads ASAP
        self._opened = False

        # wait if a read() is currently inside capture_array()
        with self._io_lock:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass

def init_input_source(input_context: InputContext) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {input_context.input}")

    if input_context.frame_rate:
        cap.set(cv2.CAP_PROP_FPS, input_context.frame_rate)

    res_map = {"sd": (640, 480), "hd": (1280, 720), "fhd": (1920, 1080)}
    if input_context.resolution: 
        w, h = res_map[input_context.resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_context.cap = cap
    input_context.width = w
    input_context.height = h
    return input_context


def init_rpicam2(width: int, height: int):
    """
    Open Raspberry Pi camera using Picamera2.

    Returns:
        PiCamera2CaptureAdapter | None:
            Camera adapter if successful, otherwise None.
    """
    try:
        from picamera2 import Picamera2
    except Exception as e:
        return None

    try:
        picam2 = Picamera2()
        fps = 30
        main = {"size": (width, height), "format": "RGB888"}
        config = picam2.create_video_configuration(main=main, controls={"FrameRate": fps})

        picam2.configure(config)
        picam2.start()


        return PiCamera2CaptureAdapter(picam2)

    except Exception as e:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
        return None
