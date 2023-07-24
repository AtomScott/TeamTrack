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
from teamtrack.utils import parse_args, make_yaml, make_video_paths, save_results, run_trackeval


def main(cfg):
    logger.info(cfg)
    tracker_name = cfg.TRACKER.NAME 
    output_dir = os.path.join(cfg.OUTPUT.ROOT, cfg.DATASET.NAME + f"-{cfg.DATASET.SUBSET}", tracker_name)
    if os.path.exists(output_dir) and not cfg.OUTPUT.OVERWRITE:
        logger.warning(f"Output directory {output_dir} already exists, exiting")
        exit(0)

    def objective(trial):
        start_time = time()
        parameters = {}
        parameters["tracker_type"] = "botsort"
        parameters["track_high_thresh"] = trial.suggest_float("track_high_thresh", 0.1, 0.9)
        parameters["track_low_thresh"] = trial.suggest_float("track_low_thresh", 0.1, parameters["track_high_thresh"])
        parameters["new_track_thresh"] = trial.suggest_float("new_track_thresh", 0.1, 0.9)
        parameters["track_buffer"] = trial.suggest_int("track_buffer", 1, 100)
        parameters["match_thresh"] = trial.suggest_float("match_thresh", 0.1, 0.2)
        parameters["cmc_method"] = "sparseOptFlow"
        parameters["proximity_thresh"] = 0.5
        parameters["appearance_thresh"] = 0.25
        parameters["with_reid"] = False
        
        parameters['imgsz'] = trial.suggest_int('imgsz', 320, 5120, step=480)
        parameters['vid_stride'] = trial.suggest_int('vid_stride', 1, 50, step=8)
        parameters['conf'] = trial.suggest_float('conf', 0.1, 0.9)

        model = YOLO(cfg.TRACKER.YOLOV8.MODEL_PATH)

        trial_name = "trial_" + str(trial.number)
        
        video_paths = make_video_paths(cfg.DATASET.ROOT, cfg.DATASET.NAME, cfg.DATASET.SUBSET)
        logger.info(f"Tracking {len(video_paths)} videos")
        for video_path in video_paths:
            seq_name = os.path.splitext(os.path.basename(video_path))[0]

            logger.info(f"Tracking {video_path}")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_yaml = make_yaml(tmpdir, parameters)
                results = model.track(source=video_path, stream=True, tracker=tmp_yaml, imgsz=parameters['imgsz'], vid_stride=parameters['vid_stride'], conf=parameters['conf'])
                output_path = os.path.join(output_dir, trial_name, "data", f"{seq_name}.txt")
                save_results(results, output_path)

        eval_config = trackeval.Evaluator.get_default_eval_config()
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}

        eval_config.update(dict(cfg.TRACKEVAL.EVAL))
        dataset_config.update(dict(cfg.TRACKEVAL.DATASET))
        dataset_config["TRACKER_SUB_FOLDER"] = os.path.join(trial_name, "data")
        dataset_config["OUTPUT_SUB_FOLDER"] = os.path.join(trial_name)
        metrics_config.update(dict(cfg.TRACKEVAL.METRICS))

        output_res, output_msg = run_trackeval(eval_config, dataset_config, metrics_config)
        logger.warning(output_res["MotChallenge2DBox"].keys())
        hota = np.mean([c['HOTA']['HOTA'] for c in output_res["MotChallenge2DBox"][tracker_name]["COMBINED_SEQ"].values()]) * 100
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
    out_path = os.path.join(output_dir, "best_params.yaml")
    logger.info(f"Saving best parameters to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f)

    target_names = ["HOTA", "time"]
    optuna.visualization.plot_pareto_front(study, target_names=target_names).write_image(os.path.join(output_dir, "pareto_front.png"))    
    
    for value_idx, target_name in enumerate(target_names):
        target = lambda t: t.values[value_idx]
        optuna.visualization.plot_optimization_history(study, target=target).write_image(os.path.join(output_dir, f"optimization_history_{target_name}.png"))
        optuna.visualization.plot_parallel_coordinate(study, target=target).write_image(os.path.join(output_dir, f"parallel_coordinate_{target_name}.png"))
        optuna.visualization.plot_param_importances(study, target=target, target_name="HOTA").write_image(os.path.join(output_dir, f"param_importances_{target_name}.png"))


if __name__ == "__main__":
    main(parse_args())

# python scripts/benchmark/yolov8/botsort/train.py --config scripts/benchmark/yolov8/botsort/config.yaml