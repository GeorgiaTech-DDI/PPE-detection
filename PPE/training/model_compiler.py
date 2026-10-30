from pathlib import Path
import argparse

from ultralytics import YOLO

"""
This script compiles an already-trained YOLO model to HEF.

# Requirements:
Compiling any model to HEF requires:
- A Python environment with the Hailo Dataflow Compiler and Ultralytics libraries installed
    - [Hailo Developer Zone Software](https://hailo.ai/developer-zone/software-downloads)
    - [Ultralytics](https://www.ultralytics.com/)
- A set of trained weights (in .pt format for YOLO models exported using the Ultralytics library)
- The dataset that was used to train the model

In this script, the images that were used in training are the same images that will be used
in calibration.

# Compilation Details
Model compilation has several stages.
1. Quantization - Weights are changed to 8 bit ints from floating point values
2. This quantized version of the model is compiled from a .har to the Hailo machine language
3. Writing to a file. Output will be a file in the format HEF.

# Arguments
This script takes in the following arguments:
- trained_weights: Required positional argument. The .pt file we are compiling.
- calibration_data: Required positional argument. The data that will be used in model calibration
- hailo-target: Optional. The target architecture. Defaults to hailo10h.
- imgsz: Optional. The size of the image. Defaults to 640.

# How to run
- Set up a virtual environment through conda or venv and install Ultralytics and the Hailo Dataflow compiler
- Run with python3 model_compiler.py path/to/weights path/to/dataset.
"""

def export_hailo_hef(
    trained_weights,
    calibration_data,
    hailo_target="hailo10h",
    imgsz=640,
):
    if not trained_weights.strip():
        raise ValueError("Enter the path to trained model weights.")
    if not calibration_data.strip():
        raise ValueError("Enter the path to the calibration dataset data.yaml.")

    best_weights = Path(trained_weights).expanduser()
    calibration_yaml = Path(calibration_data).expanduser()
    if not calibration_yaml.is_file():
        raise FileNotFoundError(f"Calibration dataset YAML not found: {calibration_yaml}")
    if not best_weights.is_file():
        raise FileNotFoundError(f"Trained model weights not found: {best_weights}")

    best_model = YOLO(str(best_weights))
    hef_path = best_model.export(
        format="hailo",
        name=hailo_target,
        data=str(calibration_yaml),
        imgsz=imgsz,
        split="train",
    )
    print(f"Hailo export output: {hef_path}")

def main():
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to a Hailo HEF model.")
    parser.add_argument("trained_weights", help="Path to the trained model weights.")
    parser.add_argument("calibration_data", help="Path to the calibration dataset data.yaml.")
    parser.add_argument("--hailo-target", default="hailo10h", help="Hailo target architecture.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for export.")
    args = parser.parse_args()

    export_hailo_hef(
        args.trained_weights,
        args.calibration_data,
        hailo_target=args.hailo_target,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
