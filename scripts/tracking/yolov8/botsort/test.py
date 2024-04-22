import argparse
import os
import tempfile

import trackeval
import numpy as np
from soccertrack.logger import logger
from teamtrack.utils import (load_yaml, make_video_paths, make_yaml,
                             parse_args, run_trackeval, save_results, visualize_results)
from ultralytics import YOLO


def main(cfg):
    logger.info(cfg)
    tracker_name = cfg.TRACKER.NAME
    output_dir = os.path.join(cfg.OUTPUT.ROOT, cfg.DATASET.NAME + f"-{cfg.DATASET.SUBSET}", tracker_name)
    if os.path.exists(output_dir) and not cfg.OUTPUT.OVERWRITE:
        logger.warning(f"Output directory {output_dir} already exists, exiting")
        exit(0)

    model = YOLO(cfg.TRACKER.YOLOV8.MODEL_PATH)
    logger.info(model)

    video_paths = make_video_paths(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.SUBSET)
    logger.info(f"Tracking {len(video_paths)} videos")
    
    
    parameters = {key.lower():value for key, value in dict(cfg.TRACKER.YOLOV8).items()}
    # read yaml file
    d = load_yaml(cfg.TRACKER.YOLOV8.CONFIGURATION_PATH)
    parameters.update(d)
    logger.info(parameters)
    
    for video_path in video_paths:
        seq_name = os.path.splitext(os.path.basename(video_path))[0]
        logger.info(f"Tracking {video_path}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_yaml = make_yaml(tmpdir, parameters)
            results = model.track(
                source=video_path,
                stream=True,
                tracker=tmp_yaml,
                imgsz=parameters["imgsz"],
                vid_stride=parameters["vid_stride"],
                conf=parameters["conf"],
            )
            output_path = os.path.join(output_dir, "data", f"{seq_name}.txt")
            save_results(results, output_path)

        if cfg.OUTPUT.SAVE_VIDEO:
            seq_name = os.path.splitext(os.path.basename(video_path))[0]
            text_path = os.path.join(output_dir, "data", f"{seq_name}.txt")
            save_path = os.path.join(output_dir, "videos", f"{seq_name}.mp4")
            logger.info(f"Visualizing tracking results to {save_path}")
            visualize_results(video_path, text_path, save_path)

    if cfg.OUTPUT.SAVE_EVAL:
        eval_config = trackeval.Evaluator.get_default_eval_config()
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
        eval_config.update(dict(cfg.TRACKEVAL.EVAL))
        dataset_config.update(dict(cfg.TRACKEVAL.DATASET))
        metrics_config.update(dict(cfg.TRACKEVAL.METRICS))
        output_res, output_msg = run_trackeval(eval_config, dataset_config, metrics_config)

            
    output_dir = cfg.OUTPUT.ROOT

if __name__ == "__main__":
    main(parse_args())
