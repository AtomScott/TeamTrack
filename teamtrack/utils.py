import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import git
import inspect
import numpy as np
import pandas as pd
import trackeval
import yaml
from sportslabkit.logger import logger
from teamtrack.config import cfg
from tqdm import tqdm
from yacs.config import CfgNode as CN

import sportslabkit as slk
from sportslabkit.mot import DeepSORTTracker, SORTTracker, BYTETracker


def get_git_root() -> Path:
    """
    Get the root of the git repository.

    Returns:
        Path: The root path of the git repository.
    """
    git_repo = git.Repo(__file__, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


def parse_args() -> CN:
    """
    Parse command line arguments.

    Returns:
        CN: Configuration node that contains all the configuration parameters.
    """
    parser = argparse.ArgumentParser(description="Parameter Search")
    parser.add_argument("--config_file", default="scripts/benchmark/yolov8/botsort.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def make_yaml(tmpdir: str, hyperparameters: dict) -> str:
    """
    Create a YAML file with hyperparameters and save it to the given directory.

    Args:
        tmpdir (str): Path to the directory where the YAML file will be saved.
        hyperparameters (dict): Hyperparameters to be saved in the YAML file.

    Returns:
        str: Path to the created YAML file.
    """
    yaml_path = os.path.join(tmpdir, "hyperparameters.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(hyperparameters, f)
    return yaml_path


def load_yaml(file_path: str) -> dict:
    """
    Load a YAML file into a Python dictionary.

    Args:
        file_path (str): Path to the YAML file to be loaded.

    Returns:
        dict: A dictionary containing the contents of the YAML file.

    Raises:
        yaml.YAMLError: If there is an error while loading the YAML file.
    """
    with open(file_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    return data


def make_video_paths(dataset_root: str, dataset_name: str, subset: str) -> list:
    """
    Return a list of paths to videos in dataset_path.

    Args:
        dataset_root (str): Root path to the dataset.
        dataset_name (str): Name of the dataset.
        subset (str): Subset of the dataset.

    Returns:
        list: List of paths to videos.

    Raises:
        ValueError: If the subset is not one of the following - 'train', 'val', 'test', 'trainval', 'all'.
    """
    # the dataset path should be pointing to a directory containing videos
    if subset in ["train", "val", "test"]:
        video_paths = sorted(glob(os.path.join(dataset_root, dataset_name, subset, "videos", "*.mp4")))
    elif subset == "trainval":
        video_paths = sorted(glob(os.path.join(dataset_root, dataset_name, "train", "videos", "*.mp4")))
        video_paths += sorted(glob(os.path.join(dataset_root, dataset_name, "val", "videos", "*.mp4")))
    elif subset == "all":
        video_paths = sorted(glob(os.path.join(dataset_root, dataset_name, "train", "videos", "*.mp4")))
        video_paths += sorted(glob(os.path.join(dataset_root, dataset_name, "val", "videos", "*.mp4")))
        video_paths += sorted(glob(os.path.join(dataset_root, dataset_name, "test", "videos", "*.mp4")))
    else:
        raise ValueError(f"Invalid subset {subset}")
    assert len(video_paths) > 0, f"No videos found in {dataset_root}/{dataset_name}/{subset}/videos"
    return video_paths


# For 'result' object, assuming a structure where 'boxes' has 'id', 'xyxy', 'conf' attributes.
def save_results(results: list, output_path: str) -> None:
    """
    Save the results of tracking a video to a file.

    Args:
        results (list): List of tracking results for a video.
        output_path (str): Path where the results will be saved.

    Returns:
        None
    """
    # the output_dir should be pointing to a directory containing the results
    rows = []
    for frame, result in enumerate(results):
        frame += 1
        n_results = len(result.boxes.id) if result.boxes.id is not None else 0

        boxes = result.boxes.xyxy.cpu().numpy().astype(int) if n_results > 0 else []
        ids = result.boxes.id.cpu().numpy().astype(int) if n_results > 0 else []
        confs = result.boxes.conf.cpu().numpy().astype(float) if n_results > 0 else []

        for box, id, conf in zip(boxes, ids, confs):
            bb_left, bb_top, bb_width, bb_height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rows.append([frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if rows == []:  # if there are no detections, create an empty file
        logger.info(f"No detections found, creating empty file {output_path}")
        with open(output_path, "w") as f:
            f.write("")
    else:
        logger.info(f"Saving results to {output_path}")
        np.savetxt(output_path, rows, fmt="%d,%d,%d,%d,%d,%d,%f,%d,%d,%d")


def run_trackeval(eval_config: dict, dataset_config: dict, metrics_config: dict) -> dict:
    """
    Run evaluation using the provided configurations.

    Args:
        eval_config (dict): Evaluation configuration.
        dataset_config (dict): Dataset configuration.
        metrics_config (dict): Metrics configuration.

    Returns:
        dict: Evaluation results.

    Raises:
        Exception: If no metrics are selected for evaluation.
    """
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception("No metrics selected for evaluation")
    return evaluator.evaluate(dataset_list, metrics_list)


def generate_colors(num_colors: int) -> list[tuple[int, int, int]]:
    """
    Generates a list of unique RGB color tuples.

    Args:
        num_colors (int): The number of unique colors to generate.

    Returns:
        list: A list of unique RGB color tuples.
    """
    color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
    return color


def visualize_results(video_path: str, text_path: str, save_path: str) -> None:
    """
    Visualize tracking results on video frames given the paths to a video file and a text file.

    Args:
        video_path (str): The path to the input video file.
        text_path (str): The path to the text file containing tracking results.
        save_path (str): The path to the output video file.

    Returns:
        None
    """
    # Load tracking results into pandas DataFrame
    columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
    data = pd.read_csv(text_path, header=None, names=columns, delimiter=",")

    # Generate unique colors for each unique ID
    unique_ids = data["id"].unique()
    colors = generate_colors(len(unique_ids))
    id_to_color = dict(zip(unique_ids, colors))

    # Create parent directory of save_path if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    frame_id = 1
    # Use tqdm for progress bar
    with tqdm(total=num_frames, ncols=70) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the detections for the current frame only
            frame_data = data[data["frame"] == frame_id]
            # Draw the detections on the frame
            for i, row in frame_data.iterrows():
                bb_left = int(row["bb_left"])
                bb_top = int(row["bb_top"])
                bb_bottom = int(row["bb_top"] + row["bb_height"])
                bb_right = int(row["bb_left"] + row["bb_width"])

                cv2.rectangle(frame, (bb_left, bb_top), (bb_right, bb_bottom), id_to_color[row["id"]], 2)
                cv2.putText(frame, f'ID: {row["id"]}', (bb_left, bb_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, id_to_color[row["id"]], 2)
            # Save the frame to the output video file
            out.write(frame)

            # Move to the next frame
            frame_id += 1
            pbar.update(1)

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def load_teamtrack_search_space(model_name):
    if model_name == "SORT":
        return {
            "self": {
                "min_length": {"type": "float", "low": 1, "high": 100},
                "max_staleness": {"type": "float", "low": 1, "high": 100},
            },
            "motion_model": {
                # "process_noise": {"type": "float", "low": 1, "high": 1e3},
                # "measurement_noise": {"type": "logfloat", "low": 1e-3, "high": 1e2},
            },
            "matching_fn": {
                "gate": {"type": "float", "low": 0.6, "high": 0.9},
            },
        }
    if model_name == "ByteTrack":
        return {
            "self": {
                "detection_score_threshold": {"type": "float", "low": 0.1, "high": 1},
            },  # should tune on SORT
            "motion_model": {},  # should tune on SORT
            "first_matching_fn": {
                "visual_metric_gate": {"type": "float", "low": 0.5, "high": 1},
                "beta": {"type": "logfloat", "low": 1e-3, "high": 1.0},
            },
            "second_matching_fn": {
                "gate": {"type": "float", "low": 0.01, "high": 1},
            },
        }
    if model_name == "DeepSORT":
        return {
            "self": {},  # should tune on SORT
            "motion_model": {},  # should tune on SORT
            "matching_fn": {
                "visual_metric_gate": {"type": "float", "low": 1e-4, "high": 1},
                "beta": {"type": "logfloat", "low": 1e-4, "high": 1},
            },
        }


def load_teamtrack_model(cfg):
    tracker_cfg = cfg.TRACKER.TEAMTRACK.TRACKER_CFG
    motion_model_cfg = cfg.TRACKER.TEAMTRACK.MOTION_MODEL
    image_model_cfg = cfg.TRACKER.TEAMTRACK.IMAGE_MODEL
    detection_model_cfg = cfg.TRACKER.TEAMTRACK.DETECTION_MODEL

    if detection_model_cfg.MODEL_NAME is not None:
        model_name = detection_model_cfg.MODEL_NAME
        _model = slk.detection_model.load(model_name=model_name)
        args = inspect.getfullargspec(_model.__init__).args
        model_cfg = {k.lower(): v for k, v in detection_model_cfg.items() if k.lower() in args}
        det_model = slk.detection_model.load(
            model_name=model_name,
            **model_cfg,
        )
        slk.logger.inspect(det_model)

    if motion_model_cfg.MODEL_NAME is not None:
        model_name = motion_model_cfg.MODEL_NAME
        _model = slk.motion_model.load(model_name=model_name)
        args = inspect.getfullargspec(_model.__init__).args
        model_cfg = {k.lower(): v for k, v in motion_model_cfg.items() if k.lower() in args}

        motion_model = slk.motion_model.load(
            model_name=model_name,
            **model_cfg,
        )
        slk.logger.inspect(motion_model)

    if image_model_cfg.MODEL_NAME is not None:
        model_name = image_model_cfg.MODEL_NAME
        _model = slk.image_model.load(model_name=model_name)
        args = inspect.getfullargspec(_model.__init__).args
        model_cfg = {k.lower(): v for k, v in image_model_cfg.items() if k.lower() in args}

        image_model = slk.image_model.load(
            model_name=model_name,
            **model_cfg,
        )
        slk.logger.inspect(image_model)

    if tracker_cfg.NAME == "SORT":
        cmm_class = getattr(slk.metrics, tracker_cfg.MATCHING_FUNCTION)
        cmm = cmm_class(use_pred_box=True)
        matching_fn = slk.matching.SimpleMatchingFunction(metric=cmm, gate=tracker_cfg.GATE)

        model = SORTTracker(
            detection_model=det_model,
            motion_model=motion_model,
            matching_fn=matching_fn,
            max_staleness=tracker_cfg.MAX_STALENESS,
            min_length=tracker_cfg.MIN_LENGTH,
        )
    elif tracker_cfg.NAME == "DeepSORT":
        matching_fn = slk.matching.MotionVisualMatchingFunction(
            motion_metric=slk.metrics.IoUCMM(use_pred_box=True),
            motion_metric_gate=tracker_cfg.MOTION_METRIC_GATE,
            visual_metric=slk.metrics.CosineCMM(),
            visual_metric_gate=tracker_cfg.VISUAL_METRIC_GATE,
            beta=tracker_cfg.BETA,
        )

        model = DeepSORTTracker(
            detection_model=det_model,
            image_model=image_model,
            motion_model=motion_model,
            matching_fn=matching_fn,
            max_staleness=tracker_cfg.MAX_STALENESS,
            min_length=tracker_cfg.MIN_LENGTH,
        )
    elif tracker_cfg.NAME == "ByteTrack":
        first_matching_fn = slk.matching.MotionVisualMatchingFunction(
            motion_metric=slk.metrics.IoUCMM(use_pred_box=True),
            motion_metric_gate=tracker_cfg.MOTION_METRIC_GATE,
            visual_metric=slk.metrics.CosineCMM(),
            visual_metric_gate=tracker_cfg.VISUAL_METRIC_GATE,
            beta=tracker_cfg.BETA,
        )

        second_matching_fn = slk.matching.SimpleMatchingFunction(
            metric=slk.metrics.IoUCMM(use_pred_box=True),
            gate=tracker_cfg.SECOND_GATE,
        )

        model = BYTETracker(
            detection_model=det_model,
            image_model=image_model,
            motion_model=motion_model,
            first_matching_fn=first_matching_fn,
            second_matching_fn=second_matching_fn,
            detection_score_threshold=tracker_cfg.DETECTION_SCORE_THRESHOLD,
            max_staleness=tracker_cfg.MAX_STALENESS,
            min_length=tracker_cfg.MIN_LENGTH,
        )
    slk.logger.inspect(model)
    return model
