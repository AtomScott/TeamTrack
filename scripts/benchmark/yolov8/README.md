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

### `Basketball_SideView`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1JJ7_fNSqZ3sZMIdj-4eYu_6OIrjA1xwo) | mAP-value-here |                     |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1bYiKG1MsOoCOrxsMR-RnysCPZUWx1Bug) | mAP-value-here |                     |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1ary2Q09Fi6edGKwD9z9Eg_hxbbECf64i) | mAP-value-here |                     |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1-4L5KuCA129WLTObCCrUuW5NF9EBoyNI) | mAP-value-here |                     |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1R-yNvyRgFPS6BaemJvfGsr0SuCfCtfR_) | mAP-value-here |                     |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1K1P0cIcWA0LspFrUeQnS28iadpP6GmQw) | mAP-value-here |                     |

### `Basketball_SideView2`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1p5RFVMvUOv3z9fqvKiwSwqMBru_pTZOV) | mAP-value-here |                     |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1lRLSemmm8l1ZQrC1RiZNwNRvyv7CSk4J) | mAP-value-here |                     |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1r8f-LkoAiOBKRVy2X6MGS8Z2xLGpaivs) | mAP-value-here |                     |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=10plWCBunYJm-TXcjtr9eT-ZiJqPQ1k4Q) | mAP-value-here |                     |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1U4Fwt1pviMmtVoYRO1Exp_gXPc3UrPpt) | mAP-value-here |                     |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1EwOmpks1UiAU38tGxSLIzPZD9k5Txvrl) | mAP-value-here |                     |

### `Basketball_TopView`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1XfgTVLFkMUBYUS7J6mGCcDfsm4uzrndB) | mAP-value-here |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1VZCQ1rkJO7C1esmtUJNGLZQNN9fV7TO9) | mAP-value-here |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1W3x0pdHRSvSxGUWTpMfjFoh23GryR-NO) | mAP-value-here |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1_KgSkfnxXwAG3U4C4ZS4MoT1Nd8WsbTa) | mAP-value-here |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1Bn_JFk2FAiFYnRGNH-GvxEmdBmC9iUP_) | mAP-value-here |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=14774yjMRsrsedM333_9eKqNSyWrMVcGx) | mAP-value-here |

### `Soccer_SideView`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1azHqR9tcAY4IJ4QIwVbbJ5-5lZAk2quZ) | mAP-value-here |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1o8UtnobJT_rcbnk_Y9kqx5tDjsrRlZNK) | mAP-value-here |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1cl55bb_Jgu8mOWednjt6Weql0GAuEe3Y) | mAP-value-here |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=156vtwzBZaPCoxq0jOfIP6ZZZ02iuPpWp) | mAP-value-here |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1PieszMIIICVP0Ntonxr_r6sVWPa17JpD) | mAP-value-here |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1hjiCAinwu5PoilVGU25AGk_bTNocf9MZ) | mAP-value-here |

### `Soccer_TopView`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1CiXe95f9rTzMXJ_NwypHiZoHIY61sEBd) | mAP-value-here |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1h_MQxd7DHBKfEFCoINQDE8f1aK9XZrye) | mAP-value-here |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1CEJTf1sDjdKH7MHJxwpULO9pK9Og0uAo) | mAP-value-here |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1z-Z9hJWqj_NDj71MuuWUj4qsPVcblx35) | mAP-value-here |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1V2FLRTEm4iCIrfP_hGfHL3M7p9DglTMT) | mAP-value-here |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1xf33gmbkBsB7-AqSO85xqnOBy3yJD5U7) | mAP-value-here |

### `Handball_SideView`

| Model        | Model Link                                                                   | mAP            | Inference Time (ms) |
| ------------ | ---------------------------------------------------------------------------- | -------------- | ------------------- |
| YOLOv8n-512  | [Download](https://drive.google.com/uc?id=1buLoXVkHY-zUUJWUzwARbGB8U2UQmboP) | mAP-value-here |
| YOLOv8n-2048 | [Download](https://drive.google.com/uc?id=1SC-EPCk61nFPSG1ya-IgFF2mYrag8LCi) | mAP-value-here |
| YOLOv8m-512  | [Download](https://drive.google.com/uc?id=1OTyzJHFy2rcs7t077znn9VjDhT2pL2YI) | mAP-value-here |
| YOLOv8m-2048 | [Download](https://drive.google.com/uc?id=1sXPr2i6FazDOyQL56VuqTjfJYlrLO3z0) | mAP-value-here |
| YOLOv8x-512  | [Download](https://drive.google.com/uc?id=1VQJky3FEyWK7rfJDfN0sLNYYKWtsVvmS) | mAP-value-here |
| YOLOv8x-2048 | [Download](https://drive.google.com/uc?id=1qiDlYCxmRCBb0ySO4ZNnVuz-q5xGcXtU) | mAP-value-here |

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