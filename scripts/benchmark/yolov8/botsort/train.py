import argparse
import os
import tempfile
from glob import glob
from time import time

import numpy as np
import optuna
import yaml
from ultralytics import YOLO

import trackeval
from soccertrack.logger import logger
from teamtrack.config import cfg
from teamtrack.utils import get_git_root


def parse_args():
    parser = argparse.ArgumentParser(description="Bot-SORT Parameter Search")
    parser.add_argument("--config_file", default="scripts/benchmark/yolov8/botsort.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def make_yaml(tmpdir, hyperparameters):
    # make a yaml file with the hyperparameters and save it to tmpdir
    yaml_path = os.path.join(tmpdir, "hyperparameters.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(hyperparameters, f)
    return yaml_path


def make_video_paths(dataset_root, dataset_name, subset):
    # return a list of paths to videos in dataset_path
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
    return video_paths


def save_results(results, output_path):
    # save the results of tracking a video to a file
    # the output_dir should be pointing to a directory containing the results
    for frame, result in enumerate(results):
        frame += 1
        # print(result.boxes)
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        ids = result.boxes.id.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy().astype(float)

        rows = []
        for box, id, conf in zip(boxes, ids, confs):
            bb_left, bb_top, bb_width, bb_height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rows.append([frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, rows, fmt="%d %d %d %d %d %d %f %d %d %d", delimiter=",")


def run_trackeval(eval_config, dataset_config, metrics_config):
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


def main(cfg):
    logger.info(cfg)

    def objective(trial):
        start_time = time()
        parameters = {}
        parameters["tracker_type"] = "botsort"
        parameters["track_high_thresh"] = trial.suggest_float("track_high_thresh", 0.1, 0.9)
        parameters["track_low_thresh"] = trial.suggest_float("track_low_thresh", 0.1, parameters["track_high_thresh"])
        parameters["new_track_thresh"] = trial.suggest_float("new_track_thresh", 0.1, 0.9)
        parameters["track_buffer"] = trial.suggest_int("track_buffer", 1, 100)
        parameters["match_thresh"] = trial.suggest_float("match_thresh", 0.1, 0.9)
        parameters["cmc_method"] = "sparseOptFlow"
        parameters["proximity_thresh"] = 0.5
        parameters["appearance_thresh"] = 0.25
        parameters["with_reid"] = False

        model = YOLO(cfg.TRACKER.YOLOV8.MODEL_PATH)

        dataset_name = cfg.DATASET.NAME
        video_paths = make_video_paths(cfg.DATASET.ROOT, dataset_name, "train")
        logger.info(f"Tracking {len(video_paths)} videos")
        for video_path in video_paths:
            seq_name = os.path.splitext(os.path.basename(video_path))[0]

            logger.info(f"Tracking {video_path}")

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_yaml = make_yaml(tmpdir, parameters)
                results = model.track(source=video_path, stream=True, tracker=tmp_yaml)
                output_path = os.path.join(cfg.OUTPUT.ROOT, dataset_name, cfg.TRACKER.YOLOV8.NAME, "data", f"{seq_name}.txt")
                save_results(results, output_path)

        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        default_metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

        eval_config = default_eval_config.update(dict(cfg.TRACKEVAL.EVAL))
        dataset_config = default_dataset_config.update(dict(cfg.TRACKEVAL.DATASET))
        metrics_config = default_metrics_config.update(dict(cfg.TRACKEVAL.METRICS))

        output_res, output_msg = run_trackeval(eval_config, dataset_config, metrics_config)
        logger.debug(f"{output_res=}")
        logger.debug(f"{output_msg=}")
        hota = output_res["HOTA"]["all"]["HOTA"]
        total_time = time() - start_time
        return hota, total_time

    sampler = optuna.samplers.NSGAIISampler()
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=cfg.HPO.N_TRIALS)

    trial_with_highest_hota = max(study.best_trials, key=lambda t: t.values[0])
    info_msg = "\nTrial with highest HOTA:\n"
    info_msg += f"\tnumber: {trial_with_highest_hota.number}\n"
    info_msg += "\tparams:\n"
    for k, v in trial_with_highest_hota.params.items():
        info_msg += f"\t\t{k}: {v}\n"
    info_msg += "\tvalues:\n"
    info_msg += f"\t\tHOTA: {trial_with_highest_hota.values[0]}\n"
    info_msg += f"\t\ttime: {trial_with_highest_hota.values[1]}\n"
    logger.info(info_msg)

    # Save the best parameters
    best_params = trial_with_highest_hota.params
    out_path = os.path.join(cfg.OUTPUT.ROOT, "best_params.yaml")
    logger.info(f"Saving best parameters to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f)

    target_names = ["HOTA", "time"]
    target = lambda t: t.values[0]
    output_dir = os.path.join(cfg.OUTPUT.ROOT, cfg.DATASET.NAME, cfg.TRACKER.NAME, "hpo")
    optuna.visualization.plot_pareto_front(study, target_names=target_names).write_html(os.path.join(output_dir, "pareto_front.html"))
    optuna.visualization.plot_optimization_history(study, target=target).write_html(os.path.join(output_dir, "optimization_history.html"))
    optuna.visualization.plot_parallel_coordinate(study, target=target).write_html(os.path.join(output_dir, "parallel_coordinate.html"))
    optuna.visualization.plot_param_importances(study, target=target, target_name="HOTA").write_html(os.path.join(output_dir, "param_importances.html"))


if __name__ == "__main__":
    main(parse_args())
