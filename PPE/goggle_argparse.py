from argparse import ArgumentParser

def parse_args():
    """
    Parse command-line arguments for the detection application.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser: ArgumentParser = ArgumentParser(
        description="Parser for PPE Detection",
    )
    parser.description = "Run object detection with optional tracking and performance measurement."

    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Enable object tracking for detections. "
            "When enabled, detected objects will be tracked across frames using a tracking algorithm "
            "(e.g., ByteTrack). This assigns consistent IDs to objects over time, enabling temporal analysis, "
            "trajectory visualization, and multi-frame association. Useful for video processing applications."
        ),
    )

    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help=(
            "Path to a text file containing class labels, one per line. "
            "Used for mapping model output indices to human-readable class names. "
            "If not specified, default labels for the model will be used (e.g., COCO labels for detection models)."
        ),
    )

    parser.add_argument(
        "--draw-trail",
        action="store_true",
        help=(
            "[Tracking only] Draw motion trails of tracked objects.\n"
            "Uses the last 30 positions from the tracker history."
        )
    )

    parser.add_argument(
        "-cr",
        "--camera-resolution",
        type=str,
        choices=["sd", "hd", "fhd"],
        help=(
            "Predefined resolution for camera input sources. "
            "Options: 'sd' (640x480, Standard Definition), 'hd' (1280x720, High Definition), "
            "'fhd' (1920x1080, Full High Definition). "
            "Default is 'sd'. This flag is only applicable when using camera input sources."
        ),
    )

    parser.add_argument(
        "--hef-path",
        "-n",
        type=str,
        default=None,
        help=(
            "Path or name of Hailo Executable Format (HEF) model file. "
            "Can be: (1) full path to .hef file, (2) model name (will search in resources), "
            "or (3) model name from available models (will auto-download if not found). "
            "If not specified, uses the default model for this application."
        ),
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help=(
            "Number of frames or images to process in parallel during inference. "
            "Higher batch sizes can improve throughput but require more memory. "
            "Default is 1 (sequential processing)."
        ),
    )

    parser.add_argument(
        "--frame-rate",
        "-f",
        type=int,
        help=(
            "Target frame rate for video processing in frames per second. "
            "Controls the playback speed and processing rate for video sources. "
            "Default is 30 FPS. Lower values reduce processing load, higher values increase throughput."
        ),
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Directory where output files will be saved. "
            "When --save-output is enabled, processed images, videos, or result files will be "
            "written to this directory. If not specified, outputs are saved to a default location "
            "or the current working directory. The directory will be created if it does not exist."
        ),
    )

    parser.add_argument(
        "--save-output",
        "-s",
        action="store_true",
        help=(
            "Enable output file saving. When enabled, processed images or videos will be saved to disk. "
            "The output location is determined by the --output-dir flag. Without this flag, output is only displayed (if applicable)."
        ),
    )

    parser.add_argument(
        "-or",
        "--output-resolution",
        nargs="+",
        type=str,
        help=(
            "Output resolution when using a camera as the input source. "
            "You can choose one of the predefined options: "
            "'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). "
            "Alternatively, specify a custom resolution in the format: "
            "--output-resolution <width> <height> (e.g., 1920 1080). "
            "This option is ignored for non-camera input sources."
        ),
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help=(
            "Enable FPS (frames per second) counter display. "
            "When enabled, the application will display real-time performance metrics "
            "showing the current processing rate. Useful for performance monitoring and optimization."
        ),
    )

    args = parser.parse_args()
    return args
