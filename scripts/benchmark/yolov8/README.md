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

| Model   | Model Link                  | Train Log Link                   | mAP            |
| ------- | --------------------------- | -------------------------------- | -------------- |
| YOLOv8n | [Download](model-link-here) | [Train Log](train-log-link-here) | mAP-value-here |
| YOLOv8s | [Download](model-link-here) | [Train Log](train-log-link-here) | mAP-value-here |
| YOLOv8m | [Download](model-link-here) | [Train Log](train-log-link-here) | mAP-value-here |
| YOLOv8l | [Download](model-link-here) | [Train Log](train-log-link-here) | mAP-value-here |
| YOLOv8x | [Download](model-link-here) | [Train Log](train-log-link-here) | mAP-value-here |

Download and use the desired model for your benchmarking.

## How to Use

This directory provides Python scripts to run each tracking algorithm on the TeamTrack dataset and output benchmarking results. The primary scripts are:

- `train.py`: Optimizes tracking algorithm parameters based on the training and validation set. The best parameters are saved in a `.yaml` file.
  
  To run `train.py`, navigate to the subdirectory for the algorithm you want to benchmark, and then run the script in the terminal:

    ```bash
    python train.py --input-dir <path_to_training_and_validation_data> --output-dir <path_to_save_yaml>
    ```

    Replace `<path_to_training_and_validation_data>` and `<path_to_save_yaml>` with your own paths.

- `evaluate.py`: Tests the model with the parameters chosen by `train.py` on the TeamTrack test set. The results are saved in a `.csv` file.
  
  After `train.py` has completed, run `evaluate.py`:

    ```bash
    python evaluate.py --input-dir <path_to_test_data_and_yaml> --output-dir <path_to_save_csv>
    ```

    Replace `<path_to_test_data_and_yaml>` and `<path_to_save_csv>` with your own paths.

The benchmarking results will be saved in a `.csv` file in the directory you specify. Please note that the results might vary depending on your machine's specifications.


## Benchmarking Results

After running the scripts, the benchmarking results will be stored in an output file within the same subdirectory. The README file in each subdirectory provides more information on understanding the output.

Please note that the results might vary depending on your machine's specifications.

Please replace `model-link-here`, `train-log-link-here`, and `mAP-value-here` with the actual data.