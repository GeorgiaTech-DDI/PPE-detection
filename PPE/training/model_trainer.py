"""Train a YOLO model from a local or Roboflow dataset."""

from __future__ import annotations
from pathlib import Path
from ultralytics import YOLO

import albumentations as A
import argparse
import os
import platform
import roboflow
import torch


MODEL_TYPES = {
    "yolo11s": "yolov11",
    "yolo11n": "yolov11",
    "yolo26s": "yolo26",
    "yolo26n": "yolo26",
}
MODEL_NAMES = tuple(MODEL_TYPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a dataset, train and validate a YOLO model, and optionally compile a Hailo HEF."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=(*MODEL_NAMES, *(f"{name}.pt" for name in MODEL_NAMES)),
        help="YOLO model family and size.",
    )

    dataset_source = parser.add_mutually_exclusive_group()
    dataset_source.add_argument(
        "--data",
        type=Path,
        help="Local Ultralytics dataset YAML file.",
    )
    dataset_source.add_argument(
        "--workspace",
        help="Roboflow workspace slug.",
    )
    parser.add_argument("--dataset", help="Roboflow project slug.")
    parser.add_argument("--version", type=int, help="Roboflow dataset version.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key; defaults to the ROBOFLOW_API_KEY environment variable.",
    )

    parser.add_argument("--epochs", type=int, default=300, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training and export image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training and validation batch size.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, mps, or a CUDA device index.")
    parser.add_argument("--workers", type=int, default=8, help="Data-loader worker processes.")
    parser.add_argument("--project", default="runs/train", help="Directory for training outputs.")
    parser.add_argument("--name", default="train", help="Name of this training run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")

    parser.add_argument("--rotate", type=float, default=0, help="Maximum rotation in degrees.")
    parser.add_argument("--shear", type=float, default=0, help="Maximum shear in degrees.")
    parser.add_argument("--mosaic", type=float, default=0, help="Mosaic probability from 0 to 1.")
    parser.add_argument("--flip-lr", type=float, default=0, help="Left/right flip probability from 0 to 1.")
    parser.add_argument("--flip-ud", type=float, default=0, help="Up/down flip probability from 0 to 1.")
    parser.add_argument("--brightness", type=float, default=0, help="Brightness adjustment strength.")
    parser.add_argument("--contrast", type=float, default=0, help="Contrast adjustment strength.")
    parser.add_argument("--noise", type=float, default=0, help="Noise strength; 0 disables noise.")
    parser.add_argument(
        "--motion-blur",
        type=int,
        nargs="?",
        const=3,
        default=0,
        help="Enable motion blur with a 3-pixel kernel, or provide a kernel size; omit to disable.",
    )

    parser.add_argument(
        "--hef-export",
        action="store_true",
        help="After validation, compile the best checkpoint to a Hailo HEF.",
    )
    parser.add_argument(
        "--hef-target",
        choices=("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l"),
        default="hailo10h",
        help="Hailo accelerator to target when --hef-export is set (default: hailo10h).",
    )

    args = parser.parse_args()
    args.model = args.model.removesuffix(".pt")
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.data is not None:
        if args.dataset or args.version is not None:
            parser.error("--data cannot be combined with Roboflow dataset options.")
        args.data = args.data.expanduser().resolve()
        if not args.data.is_file():
            parser.error(f"Dataset YAML does not exist: {args.data}")
    else:
        if not args.workspace or not args.dataset or args.version is None:
            parser.error("Use --data or provide all of --workspace, --dataset, and --version.")
        if args.version < 1:
            parser.error("--version must be a positive integer.")
        if not args.api_key:
            parser.error("Provide --api-key or set the ROBOFLOW_API_KEY environment variable.")

    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.patience < 0:
        parser.error("--patience cannot be negative.")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("--imgsz must be at least 32 and divisible by 32.")
    if args.batch < 1:
        parser.error("--batch must be positive.")
    if args.workers < 0:
        parser.error("--workers cannot be negative.")
    if args.seed < 0:
        parser.error("--seed cannot be negative.")
    if args.lr0 <= 0:
        parser.error("--lr0 must be positive.")

    for argument in ("mosaic", "flip_lr", "flip_ud"):
        value = getattr(args, argument)
        if not 0 <= value <= 1:
            parser.error(f"--{argument.replace('_', '-')} must be between 0 and 1.")
    if args.rotate < 0 or args.shear < 0:
        parser.error("--rotate and --shear cannot be negative.")
    if not 0 <= args.brightness <= 1 or not 0 <= args.contrast <= 1 or not 0 <= args.noise <= 1:
        parser.error("--brightness, --contrast, and --noise must be between 0 and 1.")
    if not 0 <= args.motion_blur <= 31:
        parser.error("--motion-blur must be between 0 and 31.")

def select_device(requested_device: str) -> str | int:
    if requested_device != "auto":
        return int(requested_device) if requested_device.isdigit() else requested_device

    if torch.cuda.is_available():
        return 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dataset(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.data is not None:
        return args.data

    datasets_dir = script_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    location = datasets_dir / f"{args.dataset}-{args.version}"
    rf = roboflow.Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.dataset)
    downloaded = project.version(args.version).download(
        MODEL_TYPES[args.model],
        location=str(location),
    )
    data_yaml = Path(downloaded.location) / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Roboflow download did not contain a dataset YAML: {data_yaml}")
    return data_yaml.resolve()


def get_augmentations(args: argparse.Namespace):

    augmentations = []
    if args.contrast:
        augmentations.append(
            A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=args.contrast,
                p=0.5,
            )
        )
    if args.noise:
        augmentations.append(A.GaussNoise(std_range=(0, args.noise), p=0.5))
    if args.motion_blur:
        kernel_size = args.motion_blur if args.motion_blur % 2 else args.motion_blur + 1
        augmentations.append(A.MotionBlur(blur_limit=(kernel_size, kernel_size), p=0.5))
    return augmentations


def train_and_validate(
    args: argparse.Namespace,
    data_yaml: Path,
    script_dir: Path,
):

    output_project = Path(args.project).expanduser()
    if not output_project.is_absolute():
        output_project = script_dir / output_project
    output_project.mkdir(parents=True, exist_ok=True)

    model = YOLO(f"{args.model}.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=select_device(args.device),
        workers=args.workers,
        project=str(output_project),
        name=args.name,
        seed=args.seed,
        lr0=args.lr0,
        degrees=args.rotate,
        shear=args.shear,
        mosaic=args.mosaic,
        fliplr=args.flip_lr,
        flipud=args.flip_ud,
        hsv_v=args.brightness,
        augmentations=get_augmentations(args),
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training did not produce the expected checkpoint: {best_weights}")
    print(f"Training outputs: {save_dir}")
    print(f"Best weights: {best_weights}")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=select_device(args.device),
        project=str(output_project),
        name=f"{args.name}-val",
    )
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    for class_index in metrics.box.ap_class_index:
        print(f"  {metrics.names[class_index]:<20} {metrics.box.maps[class_index]:.3f}")

    return best_weights, best_model


def export_hef(args: argparse.Namespace, data_yaml: Path, best_model) -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("Ultralytics Hailo HEF compilation requires Linux x86_64.")

    output_path = best_model.export(
        format="hailo",
        name=args.hef_target,
        data=str(data_yaml),
        imgsz=args.imgsz,
    )
    print(f"Hailo HEF export output: {output_path}")


def main() -> None:
    args = parse_args()
    if args.hef_export and (
        platform.system() != "Linux"
        or platform.machine().lower() not in {"x86_64", "amd64"}
    ):
        raise RuntimeError("Ultralytics Hailo HEF compilation requires Linux x86_64.")

    script_dir = Path(__file__).resolve().parent
    data_yaml = get_dataset(args, script_dir)
    _, best_model = train_and_validate(args, data_yaml, script_dir)

    if args.hef_export:
        export_hef(args, data_yaml, best_model)


if __name__ == "__main__":
    main()