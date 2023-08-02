# YOLOv8

This directory contains scripts for benchmarking YOLOv8-based multiple object tracking (MOT) algorithms on the TeamTrack dataset.

## Algorithms

The following YOLOv8-based algorithms are included:

- ByteTrack
- BotSORT

Each algorithm is in its own subdirectory:

- ByteTrack/
- BotSORT/

## Dependencies

Ensure you're working in a Python>=3.8 environment with PyTorch>=1.7 installed.

Install the `ultralytics` package using pip:

```bash
pip install ultralytics
```

## YOLOv8 Models

Download and use the desired model for your benchmarking. We don't provide training logs/scripts for these models as they can be easily replicated using the YOLOv8 training scripts.
I have the code but it's quite scrappy and I don't think it's worth sharing. If you really want it, let me know via a GitHub issue.

### `Basketball_SideView`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1JJ7_fNSqZ3sZMIdj-4eYu_6OIrjA1xwo) | 0.65      | 3.14                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1bYiKG1MsOoCOrxsMR-RnysCPZUWx1Bug) | 0.71      | 5.23                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1ary2Q09Fi6edGKwD9z9Eg_hxbbECf64i) | 0.67      | 3.50                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1-4L5KuCA129WLTObCCrUuW5NF9EBoyNI) | 0.70      | 17.22               |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1R-yNvyRgFPS6BaemJvfGsr0SuCfCtfR_) | 0.68      | 12.03               |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1K1P0cIcWA0LspFrUeQnS28iadpP6GmQw) | 0.72      | 36.56               |

### `Basketball_TopView`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1XfgTVLFkMUBYUS7J6mGCcDfsm4uzrndB) | 0.63      | 3.33                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1VZCQ1rkJO7C1esmtUJNGLZQNN9fV7TO9) | 0.56      | 7.02                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1W3x0pdHRSvSxGUWTpMfjFoh23GryR-NO) | 0.64      | 5.76                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1_KgSkfnxXwAG3U4C4ZS4MoT1Nd8WsbTa) | 0.62      | 17.20               |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1Bn_JFk2FAiFYnRGNH-GvxEmdBmC9iUP_) | 0.65      | 12.08               |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=14774yjMRsrsedM333_9eKqNSyWrMVcGx) | 0.60      | 38.50               |

### `Basketball_SideView2`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1S6xMmdN2imQC0HgwM5yY4YgFRtAr3Pm3) | 0.37      | 3.50                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1rVunUNHnSt_fOz0XoQ89D5r5i-0f_k2F) | 0.52      | 2.88                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1Zpr_TQnH9qjxkS0xQZ5SkM66Pp5S-58K) | 0.46      | 5.61                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1-MjxFtq0c8PFOwJYRJ5nCgtfRAB2KmET) | 0.60      | 14.96               |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1QTc8LQ8fo0fpAO_C1mdHg6e5gQmdML_n) | 0.46      | 8.09                |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1HgTf1nGxD4jx5v17KKoyqzUvq2B4yxSe) | 0.61      | 22.17               |

### `Soccer_TopView`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1azHqR9tcAY4IJ4QIwVbbJ5-5lZAk2quZ) | 0.21      | 2.71                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1o8UtnobJT_rcbnk_Y9kqx5tDjsrRlZNK) | 0.24      | 8.40                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1cl55bb_Jgu8mOWednjt6Weql0GAuEe3Y) | 0.17      | 3.41                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=156vtwzBZaPCoxq0jOfIP6ZZZ02iuPpWp) | 0.23      | 17.25               |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1PieszMIIICVP0Ntonxr_r6sVWPa17JpD) | 0.19      | 12.08               |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1hjiCAinwu5PoilVGU25AGk_bTNocf9MZ) | 0.22      | 101.61              |

### `Soccer_SideView`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1azHqR9tcAY4IJ4QIwVbbJ5-5lZAk2quZ) | 0.18      | 3.04                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1o8UtnobJT_rcbnk_Y9kqx5tDjsrRlZNK) | 0.53      | 4.39                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1cl55bb_Jgu8mOWednjt6Weql0GAuEe3Y) | 0.20      | 3.22                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=156vtwzBZaPCoxq0jOfIP6ZZZ02iuPpWp) | 0.52      | 7.04                |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1PieszMIIICVP0Ntonxr_r6sVWPa17JpD) | 0.20      | 7.75                |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1hjiCAinwu5PoilVGU25AGk_bTNocf9MZ) | 0.55      | 32.10               |

### `Handball_SideView`

| Model        | Model Link                                                                   | mAP@50:95 | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | --------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1JJ7_fNSqZ3sZMIdj-4eYu_6OIrjA1xwo) | 0.41      | 3.50                |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1bYiKG1MsOoCOrxsMR-RnysCPZUWx1Bug) | 0.66      | 3.67                |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1ary2Q09Fi6edGKwD9z9Eg_hxbbECf64i) | 0.50      | 4.63                |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1-4L5KuCA129WLTObCCrUuW5NF9EBoyNI) | 0.68      | 6.94                |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1R-yNvyRgFPS6BaemJvfGsr0SuCfCtfR_) | 0.53      | 7.75                |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1K1P0cIcWA0LspFrUeQnS28iadpP6GmQw) | 0.69      | 26.61               |

## How to Use

This directory provides Python scripts to run each tracking algorithm on the TeamTrack dataset and output benchmarking results. You can configure the script by changing the parameters in the `config.yaml` file or via the command line arguments. Please refer to the README file in each subdirectory for more information.
The primary scripts are:

- `hpo.py`: Optimizes tracking algorithm parameters based on the training and validation set. The best parameters are saved in a `.yaml` file under the directory `outputs/benchmark/hpo/dataset_name/tracker_name`  (default).
  
  To run `hpo.py`, navigate to the subdirectory for the algorithm you want to benchmark, and then run the script in the terminal:

    ```bash
    python scripts/benchmark/yolov8/botsort/hpo.py --config scripts/benchmark/yolov8/botsort/config.yaml
    ```

- `test.py`: Tests the tracking algorithm on the test set. The results are saved in a under the directory `outputs/benchmark/test/dataset_name/tracker_name` (default).

    ```bash
    python scripts/benchmark/yolov8/botsort/test.py --config scripts/benchmark/yolov8/botsort/test.yaml
    ```

## Benchmarking Results

After running the scripts, the benchmarking results will be stored in an output file within the same subdirectory. The README file in each subdirectory provides more information on understanding the output.

Please note that the results might vary depending on your machine's specifications.
