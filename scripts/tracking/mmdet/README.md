# MM-Detection

This directory contains scripts for benchmarking MM-Detection-based multiple object tracking (MOT) algorithms on the TeamTrack dataset.

## Algorithms

The following MM-Detection-based algorithms are included:

- SORT
- DeepSORT
- ByteTrack
- StrongSORT
- OCSORT
- QDTrack

Each algorithm is in its own subdirectory under MMDET:

- MMDET/SORT/
- MMDET/DeepSORT/
- MMDET/ByteTrack/
- MMDET/StrongSORT/
- MMDET/OCSORT/
- MMDET/QDTrack/

## Dependencies

First, install MMEngine and MMCV using MIM:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

Then, install MMDetection:

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e . -r requirements/tracking.txt
```

Finally, install TrackEval:

```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

See the [MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html) for more information.

## MM-Detection Detection Models

Download and use the desired model for your benchmarking. We made YOLO-x and Faster R-CNN models available for download. The models are trained on the COCO dataset. Unfortunately, we were only able to train at size YOLO-x and Faster-RCNN-r50-FPN at image size 2048 due to time constraints. These models were chose to reflect the original paper's implementation of the tracking algorithms. YOLOX-x was chosen to because it is presumably the best performing YOLO model. The image size (2048) was matched to YOLOv8 to ensure a fair comparison.

### `Soccer_SideView`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 6.6       | 129.6               |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |

### `Soccer_TopView`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 10.0      | 129.9               |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |

### `Basketball_SideView`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 44.40     | 135.3               |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |

### `Basketball_TopView`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 56.20     | 138.0               |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |

### `Basketball_SideView2`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 38.70     | 180.9               |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |

### `Handball_SideView`

| Model               | Model Link    | mAP@50:95 | Inference Time (ms) |
| ------------------- | ------------- | --------- | ------------------- |
| Faster-RCNN-r50-FPN | [Download](#) | 33.6      | 71.9                |
| YOLOX-x             | [Download](#) |           | mAP-value-here      |


## How to Use

This directory provides Python scripts to run each tracking algorithm on the TeamTrack dataset and output benchmarking results. The primary scripts are:

- `train.py`: Optimizes tracking algorithm parameters based on the training and validation set. The best parameters are saved in a .yaml file.

To run `train.py`, navigate to the subdirectory for the algorithm you want to benchmark, and then run the script in the terminal:

```bash
python train.py --input-dir <path_to_training_and_validation_data> --output-dir <path_to_save_yaml>
```

Replace `<path_to_training_and_validation_data>` and `<path_to_save_yaml>` with your own paths.

- `test.py`: Tests the model with the parameters chosen by `train.py` on the TeamTrack test set. The results are saved in a .csv file.

After `train.py` has completed, run `test.py`:

```bash
python test.py --input-dir <path_to_test_data_and_yaml> --output-dir <path_to_save_csv>
```

Replace `<path_to_test_data_and_yaml>` and `<path_to_save_csv>` with your own paths.

The benchmarking results will be saved in a .csv file in the directory you specify. Please note that the results might vary depending on your machine's specifications.

## Benchmarking Results

After running the scripts, the benchmarking results will be stored in an output file within the same subdirectory. The README file in each subdirectory provides more information on understanding the output.

Please note that the results might vary depending on your machine's specifications.

Please replace model-link-here, train-log-link-here, and mAP-value-here with the actual data.
