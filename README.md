
# TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos
> This is a work in progress. The code and dataset will be fully released soon. Stay tuned!
### <a href="https://atomscott.github.io/TeamTrack/" target="_blank">Project</a> | <a href="" target="_blank">Paper</a> | <a href="" target="_blank">Supplementary</a> | <a href="" target="_blank">Arxiv</a> <br>
This repository contains the source code and the official benchmark dataset for the paper "TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos" by [Atom Scott](https://twitter.com/AtomJamesScott), et al.

![](https://raw.githubusercontent.com/AtomScott/TeamTrack/gh-pages/static/images/banner_image.png)

___


## Introduction

TeamTrack presents a new benchmark dataset and a novel algorithm for multi-object tracking (MOT) in team sports. The challenge of object occlusions, similar appearances, and complex movements inherent in team sports necessitated the development of a robust dataset and an algorithm for MOT. The dataset includes full-pitch videos from soccer, basketball, and handball games, captured using fisheye and drone cameras, and contains over 4 million annotated bounding boxes. 

The algorithm introduced in this paper incorporates trajectory forecasting using a graph neural network (GNN) to model complex group movement patterns in MOT.

## Dataset

The TeamTrack dataset features high-resolution (4K to 8K) full-pitch videos from soccer, basketball, and handball games. It contains over 4 million annotated bounding boxes with unique IDs and serves as an extensive benchmark dataset for multi-object tracking in team sports.

You can download the TeamTrack dataset from either [Google Drive link](https://drive.google.com/drive/u/1/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ) or [Kaggle](https://www.kaggle.com/datasets/atomscott/teamtrack).

The Google Drive contains three .zip files:
- `teamtrack-mot.zip`: the TeamTrack tracking data formatted in MOT Challenge style. Refer to the [MOT Challenge official docs](https://github.com/JonathonLuiten/TrackEval/tree/master/docs/MOTChallenge-Official) for more information on this format.
- `teamtrack.zip`: the TeamTrack tracking data formatted in SoccerTrack style.
- `teamtrack-trajectory.zip`: the TeamTrack tracking data projected to pitch coordinates for use as trajectory data.

Each .zip file contains train and validation splits.

Once downloaded, unzip the files. The unzipped directory for `teamtrack` and `teamtrack-trajectory` should look like:

```
📁 {teamtrack, teamtrack-trajectory}/
├─📁 Basketball_SideView/
├─📁 Basketball_SideView2/
├─📁 Basketball_TopView/
├─📁 Handball_SideView/
├─📁 Soccer_SideView/
└─📁 Soccer_TopView/
```

For `teamtrack`, each dataset will contain:

```
📁 {dataset}/
├─📁 test/
│ ├─📁 annotations/ .csv files
│ └─📁 videos/ .mp4 files
├─📁 train/
└─📁 val/
```

For `teamtrack-trajectory`, each dataset will contain:

```
📁 {dataset}/
├─📁 test/ .txt files
├─📁 train/
└─📁 val/
```

Please ensure that your local copy of the dataset matches this structure before running the experiments.


## Getting Started

Before running the experiments, ensure you have installed the necessary dependencies. Clone this repository and set up the environment by following these steps:

```
git clone https://github.com/atomscott/teamtrack.git
cd teamtrack
pip install -r requirements.txt
```

This project depends on a two other repos of mine:

* [SportsLabKit](https://github.com/AtomScott/SportsLabKit).
* [TeamTraj](https://github.com/AtomScott/TeamTraj).

## Scripts

In the scripts directory, you can find code to reproduce the following experiments described in the paper:

1. **Object Detection:** The fine-tuning process for the YOLOv8 model on the TeamTrack dataset.

2. **Trajectory Forecasting:** Training and evaluating the trajectory forecasting models (Constant, LSTM, GNN) on the TeamTrack dataset.

3. **Multi-Object Tracking:** Scripts to run multiple object tracking.

```
📁 scripts/
├─📁 tracking/    # scripts to run benchmarks on detection and tracking
├─📁 forecasting/ # scripts to run benchmarks on trajectory forecasting
└─📁 preproc/     # scripts to run preprocessing on data
```

## Notebooks

Notebooks contatining detailed analysis of the TeamTrack dataset is also provided in the 'notebooks' directory. It includes metrics such as IoU on adjacent frames, frequency of relative position switches, cosine distances of re-identification features, and others.

## Citation

If you find our work useful for your research, please consider citing:

```
BibTeX Entry Goes Here (will update in the future)
```
We're excited to see the innovative ways this dataset and code will be utilized in the future.
