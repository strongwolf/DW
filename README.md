# A Dual Weighting Label Assignment Scheme for Object Detection
This repo hosts the code for implementing the DW, as presented in our CVPR 2022 paper.

## Introduction

Label assignment (LA), which aims to assign each training sample a positive (pos) and a negative (neg) loss weight, plays an important role in object detection. Existing LA methods mostly focus on the design of pos weighting function, while the neg weight is directly derived from the pos weight. Such a mechanism limits the learning capacity of detectors. In this paper, we explore a new weighting paradigm, termed  dual weighting (DW), to specify pos and neg weights separately. We first identify the key influential factors of pos/neg weights by analyzing the evaluation metrics in object detection, and then design the pos and neg weighting functions based on them. Specifically, the pos weight of a sample is determined by the consistency degree between its classification and localization scores, while the neg weight is decomposed into two terms: the probability that it is a neg sample and its importance conditioned on being a neg sample.  Such a weighting strategy offers greater flexibility to distinguish between important and less important samples, resulting in a more effective object detector. Equipped with the proposed DW method, a single FCOS-ResNet-50 detector can reach **41.5** mAP on COCO under 1x schedule, outperforming other existing LA methods. It consistently improves the baselines on COCO by a large margin under various backbones without bells and whistles.

## Installation

- This DW implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Therefore the installation is the same as original MMDetection.

- Please check [get_started.md](docs/get_started.md) for installation. Make sure the version of MMDection is larger than 2.18.0.

## Results and Models

For your convenience, we provide the following trained models. These models are trained with a mini-batch size of 16 images on 8 Nvidia 3090Ti GPUs (2 images per GPU).

| Backbone     | Style     | DCN     | MS <br> train | Box refine | Lr <br> schd | box AP <br> (val)  | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:---------:|:-------:|:-------------:|:----------:|:------------:|:-------------------:|:--------------------------------------:|
| R-50         | pytorch   | N       | N             | N          | 1x           |                |  [model] &#124; [log]|
| R-50         | pytorch   | N       | N             | Y          | 1x           |                |  [model] &#124; [log]|
| R-50         | pytorch   | N       | Y             | Y          | 2x           |                |  [model] &#124; [log]|
| R-50         | pytorch   | Y       | Y             | Y          | 2x           |                |  [model] &#124; [log]|
| R-101        | pytorch   | N       | N             | Y          | 1x           |                |  [model] &#124; [log]|

**Notes:**

- The MS-train maximum scale range is 1333x[480:960] (`range` mode) and the inference scale keeps 1333x800.
- DCN means using `DCNv2` in both backbone and head.

## Inference

Assuming you have put the COCO dataset into `data/coco/` and have downloaded the models into the `weights/`, you can now evaluate the models on the COCO val2017 split:

```
bash dist_test.sh configs/dw_r50_fpn_1x_coco.py weights/r50_1x.pth 8 --eval bbox
```

## Training

The following command line will train `dw_r50_fpn_1x_coco` on 8 GPUs:

```
bash dist_train.sh configs/dw_r50_fpn_1x_coco.py 8 --work-dir weights/r50_1x
```