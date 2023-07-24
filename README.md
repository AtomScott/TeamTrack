
# TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos
> This is a work in progress. The code and dataset will be fully released soon. Stay tuned!
### <a href="https://atomscott.github.io/TeamTrack/" target="_blank">Project</a> | <a href="" target="_blank">Paper</a> | <a href="" target="_blank">Supplementary</a> | <a href="" target="_blank">Arxiv</a> <br>
This repository contains the source code and the official benchmark dataset for the paper "TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos" by [Atom Scott](https://twitter.com/AtomJamesScott), et al.

![](https://raw.githubusercontent.com/AtomScott/TeamTrack/gh-pages/static/images/banner_image.png)


> **TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos**<br>
> Atom Scott, Ikuma Uchida, Ning Ding, Rikuhei Umemoto, Rory Bunker, Ren Kobayashi, Takeshi Koyama, Masaki Onishi, Yoshinari Kameda, Keisuke Fujii<br>
> <a href="http://arxiv.org/abs/xxx" target="_blank">http://arxiv.org/abs/xxx </a> <br>
>
>**Abstract:** This paper presents TeamTrack, a new benchmark dataset and algorithm for multi-object tracking (MOT) in team sports, specifically in full-pitch videos. MOT in team sports is a challenging task due to object occlusions, similar appearances, and complex movements. Existing methods often struggle to accurately track objects in these scenarios. To address this challenge, the proposed TeamTrack dataset captures diverse object appearances and movements in soccer, basketball, and handball games using high-resolution fisheye and drone cameras. The dataset includes over 4 million annotated bounding boxes and provides a comprehensive resource for developing and evaluating MOT algorithms. The paper also introduces a new MOT approach that incorporates trajectory forecasting using a graph neural network (GNN) to model complex group movement patterns. The experiments demonstrate the effectiveness of the proposed algorithm on the TeamTrack dataset.



___


## Table of Contents

- [TeamTrack: An Algorithm and Benchmark Dataset for Multi-Sport Multi-Object Tracking in Full-pitch Videos](#teamtrack-an-algorithm-and-benchmark-dataset-for-multi-sport-multi-object-tracking-in-full-pitch-videos)
    - [Project | Paper | Supplementary | Arxiv ](#project--paper--supplementary--arxiv-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Getting Started](#getting-started)
  - [Experiments](#experiments)
  - [Performance Evaluation](#performance-evaluation)
  - [Comparison with Baselines](#comparison-with-baselines)
  - [Model Interpretation](#model-interpretation)
  - [Dataset Analysis](#dataset-analysis)
  - [Support](#support)
  - [Citation](#citation)

## Introduction

TeamTrack presents a new benchmark dataset and a novel algorithm for multi-object tracking (MOT) in team sports. The challenge of object occlusions, similar appearances, and complex movements inherent in team sports necessitated the development of a robust dataset and an algorithm for MOT. The dataset includes full-pitch videos from soccer, basketball, and handball games, captured using fisheye and drone cameras, and contains over 4 million annotated bounding boxes. 

The algorithm introduced in this paper incorporates trajectory forecasting using a graph neural network (GNN) to model complex group movement patterns in MOT.

That's a valuable addition. Here's how you could modify the "Dataset" section:

## Dataset

The TeamTrack dataset features high-resolution (4K to 8K) full-pitch videos from soccer, basketball, and handball games. It contains over 4 million annotated bounding boxes with unique IDs and serves as an extensive benchmark dataset for multi-object tracking in team sports.

You can download the TeamTrack dataset from this [Google Drive link](https://drive.google.com/drive/u/1/folders/1D3jxrEWgWke0l1TWC_052OhYVs2IwDVZ).

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

Also, you will need to add submodules to the external directory. To do this, run the following commands:

```
git submodule add https://github.com/JonathonLuiten/TrackEval.git external/TrackEval
```

This project depends on the SoccerTrack repository. You can find it [here](https://github.com/AtomScott/SoccerTrack).

Pre-trained models or weights required for the experiments can be found [here](https://bit.ly/3NYaMWG).


## Experiments

This repository contains code to reproduce the following experiments described in the paper:

1. **Object Detection:** The fine-tuning process for the YOLOv8 model on the TeamTrack dataset.

2. **Trajectory Forecasting:** Training and evaluating the trajectory forecasting models (Constant, LSTM, GNN) on the TeamTrack dataset.

3. **Multi-Object Tracking:** Implementation of the tracking pipeline, including the data association model.

Each experiment is located in its respective directory under the 'experiments' directory. For example, to reproduce the object detection experiment, navigate to the 'object_detection' directory:

```
cd experiments/object_detection
python train.py
```

Ensure to replace 'object_detection' and 'train.py' with the correct experiment directory and script name respectively.

## Performance Evaluation

Results from a comprehensive performance evaluation of the object detection, trajectory forecasting, and multi-object tracking models on the TeamTrack dataset are available in the 'results' directory. Metrics such as mAP, RMSE, HOTA, MOTA, IDF1, DetA, AssA are used

 to evaluate the models.

## Comparison with Baselines

Comparison of the proposed models with baseline methods (DeepSORT, ByteTrack) is also available in the 'comparison' directory. 

## Model Interpretation

Interpretation of the models' behavior and characteristics are available in the 'interpretation' directory. This includes visualization of the trajectories predicted by the trajectory forecasting model, analysis of the impact of different model components on the overall performance, and identification of the models' limitations or shortcomings.

## Dataset Analysis

A detailed analysis of the TeamTrack dataset is also provided in the 'analysis' directory. It includes metrics such as IoU on adjacent frames, frequency of relative position switches, cosine distances of re-identification features, and others.

## Support

For any queries or issues with the code or dataset, please raise an issue in the GitHub repository.

## Citation

If you find our work useful for your research, please consider citing:

```
BibTeX Entry Goes Here
```
We're excited to see the innovative ways this dataset and code will be utilized in the future.
