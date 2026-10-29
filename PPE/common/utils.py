import cv2
import numpy as np

def letterbox_img(image: np.ndarray,
                  size: int = 640,
                  fill: int = 0,
                  resize_interpolation: int = cv2.INTER_CUBIC):
    """
    Letterboxes an image to the model input size, preserving aspect ratio.
    """
    h, w = image.shape[:2]

    scale = min(size / w, size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=resize_interpolation)

    pad_x = size - new_w
    pad_y = size - new_h

    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(fill, fill, fill)
    )

    return np.ascontiguousarray(image)
