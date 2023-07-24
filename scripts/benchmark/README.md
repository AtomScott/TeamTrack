# TeamTrack Benchmarking Scripts

This repository contains scripts for benchmarking various multiple object tracking (MOT) algorithms on the TeamTrack dataset. For more information on TeamTrack, please refer to the main documentation.

The repository is organized as follows:

``` bash
- TeamTrack/
- Yolov8/
  - ByteTrack/
  - BotSORT/
- MMDET/
   - SORT/
   - DeepSORT/
   - ByteTrack/
   - StrongSORT/
   - OCSORT/
   - QDTrack/
- Other/
   - MOTR/
   - MOTRv2/
   - TransTrack/
   - TrackFormer/
   - QuoVadis/
```

In each directory, there is a detailed README file that contains the step-by-step guide on how to run the corresponding algorithm, as well as the requirements for that algorithm.

## Data Download

The TeamTrack dataset is available for download from this [Google Drive link](insert-link-here). The Google Drive contains two .zip files:

- `teamtrack-mot.zip`: The TeamTrack tracking data formatted in MOT Challenge style. Refer to the [MOT Challenge official docs](insert-link-here) for more information on this format.
- `teamtrack.zip`: The TeamTrack tracking data formatted in SoccerTrack style.

## How to Use

1. Download and unzip the desired dataset from the Google Drive link.
2. Choose the algorithm you want to benchmark.
3. Navigate into the corresponding directory.
4. Follow the instructions in the README file within the directory.

## Benchmarking Results

After running the scripts, the benchmarking results will be stored in an output file. The README file in each directory provides more information on understanding the output.

We use [TrackEval](https://github.com/JonathonLuiten/TrackEval/tree/master) to evaluate the tracking results.
Clone the respository to the external directory and add the path to the `PYTHONPATH` environment variable.

Requirements
Code tested on Python 3.7.

Minimum requirements: numpy, scipy
For plotting: matplotlib
For segmentation datasets (KITTI MOTS, MOTS-Challenge, DAVIS, YouTube-VIS): pycocotools
For DAVIS dataset: Pillow
For J & F metric: opencv_python, scikit_image
For simples test-cases for metrics: pytest
use pip3 -r install requirements.txt to install all possible requirements.

use pip3 -r install minimum_requirments.txt to only install the minimum if you don't need the extra functionality as listed above.

