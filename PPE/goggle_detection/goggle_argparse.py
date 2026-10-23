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
    parser.description = "Run object detection."

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
