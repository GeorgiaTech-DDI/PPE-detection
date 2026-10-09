import numpy as np
import cv2
from hailo_platform import VDevice

timeout_ms = 1000


# Takes whatever size input, scales to 640x640 with letterboxing.
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
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return np.ascontiguousarray(image), scale, left, top


with VDevice() as vdevice:
    print("VDevice created successfully")

    # Load HEF model
    infer_model = vdevice.create_infer_model("scrfd_10g.hef")

    print(f"Model loaded: input shape {infer_model.input().shape}")

    print("\nOutput shapes are:")
    for name in infer_model.output_names:
        print(f"{name}: {infer_model.output(name).shape}")

    # Preprocess image
    img, scale, pad_left, pad_top = preprocess(
        "./hide_the_pain_harold.jpg"
    )

    print("\nPreprocessing:")
    print("scale:", scale)
    print("pad_left:", pad_left)
    print("pad_top:", pad_top)
    print("input shape:", img.shape)
    print("input dtype:", img.dtype)

    # Configure model
    with infer_model.configure() as configured_infer_model:
        print("\nModel configured")

        bindings = configured_infer_model.create_bindings()

        # Set input
        bindings.input().set_buffer(img)

        # Allocate output buffers
        for name in infer_model.output_names:
            output_shape = infer_model.output(name).shape

            output_buffer = np.zeros(
                output_shape,
                dtype=np.uint8
            )

            bindings.output(name).set_buffer(output_buffer)

        # Run inference
        print("\nRunning synchronous inference...")

        configured_infer_model.run(
            [bindings],
            timeout_ms
        )

        print("Inference finished")

        # Store all outputs
        outputs = {}

        print("\n==============================")
        print("RAW OUTPUT INFORMATION")
        print("==============================")

        for name in infer_model.output_names:
            output = bindings.output(name).get_buffer()

            outputs[name] = output

            print(f"\n{name}")
            print("shape:", output.shape)
            print("dtype:", output.dtype)
            print("min:", output.min())
            print("max:", output.max())
            print("mean:", output.mean())
            print("first 10 values:", output.flatten()[:10])

        print("\n==============================")
        print("FINISHED")
        print("==============================")