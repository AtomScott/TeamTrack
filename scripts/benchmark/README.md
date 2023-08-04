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

### Handball_SideView-test

| Methods          |   HOTA |   DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | -----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 75.079 | 75.491 | 74.696 | 91.554 |   96 |  89.68 | 83.771 | 99.322 | 92.255 |  146 |
| yolov8-bytetrack | 73.528 | 73.849 | 73.241 | 89.389 |  293 | 87.582 | 83.566 | 97.129 | 92.335 |  144 |

### Basketball_SideView2-test

| Methods          |   HOTA |   DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | -----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 47.266 | 67.593 |   33.1 | 80.204 |   57 | 50.782 | 84.355 | 98.972 | 81.626 |    8 |
| yolov8-bytetrack |  42.87 | 54.655 | 33.656 | 64.958 |   46 | 53.574 | 84.382 | 99.678 | 65.634 |    2 |

### Basketball_TopView-test

| Methods          |   HOTA |  DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | ----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 66.292 | 62.66 | 70.307 | 88.999 |    3 |  93.92 | 71.784 | 98.999 | 89.912 |   78 |
| yolov8-bytetrack | 65.702 | 65.12 | 66.444 | 89.624 |   42 | 92.048 | 71.754 | 96.288 |  93.28 |   79 |

### Soccer_TopView-test

| Methods          |   HOTA |   DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | -----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 51.933 | 51.065 | 53.259 | 42.706 |  279 | 65.712 | 64.658 | 71.615 | 70.995 |  112 |
| yolov8-bytetrack | 53.709 | 51.388 | 56.495 | 43.344 |  106 | 69.184 | 64.854 | 71.977 | 71.071 |  108 |

### Soccer_SideView-test

| Methods          |   HOTA |   DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | -----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 58.419 | 62.808 | 54.524 | 84.204 |  598 | 73.765 | 75.965 | 98.807 | 85.584 |  172 |
| yolov8-bytetrack | 59.281 | 64.443 | 54.745 | 86.417 |  564 | 74.234 | 75.957 | 98.676 | 87.923 |  183 |

### Basketball_SideView-test

| Methods          |   HOTA |   DetA |   AssA |   MOTA | IDSW |   IDF1 |   MOTP | CLR_Pr | CLR_Re |   MT |
| :--------------- | -----: | -----: | -----: | -----: | ---: | -----: | -----: | -----: | -----: | ---: |
| yolov8-botsort   | 75.195 | 79.219 |  71.44 | 94.332 |  149 | 85.947 | 84.617 |  99.24 | 95.311 |   70 |
| yolov8-bytetrack | 76.198 | 75.461 | 76.948 | 89.322 |   25 | 88.622 | 85.379 | 99.961 | 89.399 |   69 |

The symbol * indicates that the results are obtained from my own implementation of the algorithm. For other algorithms, the implementation source is specified by the prefix, e.g. `yolov8-` indicates that the results are obtained from the Yolov8 implementation.

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

