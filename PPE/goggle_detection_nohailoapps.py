from typing import Optional

from goggle_argparse import parse_args
import logging
import json
from common.data_classes import *
from common.tools import init_input_source

logger = logging.getLogger("google_detection_logger")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

def run_inference_pipeline(
    network_path: str,
    labels: list[str],
    input_context: InputContext,
    visualization_settings: Optional[VisualizationSettings]=None,
    enable_tracking: bool=False,
    show_fps: bool=False,
    draw_trail: bool=False,
):
    try:
        with open("config.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {
            "visualization_params": {
                "score_thres": 0.15,
                "max_boxes_to_draw": 500,
                "tracker": {
                "track_thresh": 0.1,
                "track_buffer": 30,
                "match_thresh": 0.9,
                "aspect_ratio_thresh": 2.0,
                "min_box_area": 500,
                "mot20": False
                }
            }
        }

    input_context = init_input_source(input_context)

def main() -> None:
    """
    Main function to run the script.
    """
    args = parse_args()
    # init_logging(level=level_from_args(args))
    # handle_and_resolve_args(args, APP_NAME)
    # input_context = InputContext(
    #     input_src=args.input,
    #     batch_size=args.batch_size,
    #     resolution=args.camera_resolution,
    #     frame_rate=args.frame_rate,
    # )
    # visualization_settings = VisualizationSettings(
    #     output_dir=args.output_dir,
    #     save_stream_output=args.save_output,
    #     output_resolution=args.output_resolution,
    # )
    # run_inference_pipeline(
    #     args.hef_path,
    #     args.labels,
    #     input_context,
    #     visualization_settings,
    #     args.track,
    #     args.show_fps,
    #     args.draw_trail
    # )