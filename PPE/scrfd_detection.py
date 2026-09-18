import numpy as np
import cv2
from hailo_platform import VDevice

timeout_ms = 1000


# Vibecoded. Do not reuse except in testing or if HEAVILY reviewed.
# Takes whatever size input, scales to 640x640 with letterboxes.
# Returns Numpy ndarray
def preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    scale = min(640 / w, 640 / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    image = cv2.resize(image, (new_w, new_h))

    pad_x = 640 - new_w
    pad_y = 640 - new_h

    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top

    image = cv2.copyMakeBorder(
        image,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return np.ascontiguousarray(image)


# The vdevice is used as a context manager ("with" statement) to ensure it's released on time.
with VDevice() as vdevice:
    print("VDevice created successfully")

    # Create an infer model from an HEF:
    infer_model = vdevice.create_infer_model('')
    print(f"Model loaded: input shape {infer_model.input().shape}, output shape {infer_model.output().shape}")

    img = preprocess("./hide_the_pain_harold.jpg")

    # Configure the infer model and create bindings for it
    with infer_model.configure() as configured_infer_model:
        print("Model configured")
        bindings = configured_infer_model.create_bindings()

        # Set input and output buffers
        buffer = np.zeros(infer_model.input().shape, dtype=np.uint8)
        bindings.input().set_buffer(img)

        buffer = np.zeros(infer_model.output().shape, dtype=np.uint8)
        bindings.output().set_buffer(buffer)

        # Run synchronous inference and access the output buffers
        print("Running synchronous inference...")
        configured_infer_model.run([bindings], timeout_ms)
        buffer = bindings.output().get_buffer()
        print(f"Synchronous inference done - output shape: {buffer.shape}")