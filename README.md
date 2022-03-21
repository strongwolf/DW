# A Dual Weighting Label Assignment Scheme for Object Detection
This repo hosts the code for implementing the [DW](https://arxiv.org/pdf/2203.09730.pdf), as presented in our CVPR 2022 paper.

## Introduction

Label assignment (LA), which aims to assign each training sample a positive (pos) and a negative (neg) loss weight, plays an important role in object detection. Existing LA methods mostly focus on the design of pos weighting function, while the neg weight is directly derived from the pos weight. Such a mechanism limits the learning capacity of detectors. In this paper, we explore a new weighting paradigm, termed  dual weighting (DW), to specify pos and neg weights separately. We first identify the key influential factors of pos/neg weights by analyzing the evaluation metrics in object detection, and then design the pos and neg weighting functions based on them. Specifically, the pos weight of a sample is determined by the consistency degree between its classification and localization scores, while the neg weight is decomposed into two terms: the probability that it is a neg sample and its importance conditioned on being a neg sample.  Such a weighting strategy offers greater flexibility to distinguish between important and less important samples, resulting in a more effective object detector. Equipped with the proposed DW method, a single FCOS-ResNet-50 detector can reach **41.5** mAP on COCO under 1x schedule, outperforming other existing LA methods. It consistently improves the baselines on COCO by a large margin under various backbones without bells and whistles.

## Installation

- This DW implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Therefore the installation is the same as original MMDetection.

- Please check [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation. Make sure the version of MMDetection is larger than 2.18.0.

## Results and Models

For your convenience, we provide the following trained models. These models are trained with a mini-batch size of 16 images on 8 Nvidia RTX 3090 GPUs (2 images per GPU).

| Backbone     | Style     | DCN     | MS <br> train | Box refine | Lr <br> schd | box AP <br> (val)  | &nbsp; &nbsp; Download  &nbsp; &nbsp;  |
|:------------:|:---------:|:-------:|:-------------:|:----------:|:------------:|:-------------------:|:--------------------------------------:|
| R-50         | pytorch   | N       | N             | N          | 1x           | 41.5               |  [model](https://drive.google.com/file/d/1pcftxE1fUzHxWFVPNbqYTmJZlmevZ66U/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1UhKctdYbKcwKkN9BNRFdorf1WY-hk5QS/view?usp=sharing)|
| R-50         | pytorch   | N       | N             | Y          | 1x           | 42.1               |  [model](https://drive.google.com/file/d/1Vzml7u5bQTA_qYLu826vj2HwFUXcAOeL/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1fQkp48N3KsLgQSMN8nRmQJ15VWRXpYTT/view?usp=sharing)|
| R-50         | pytorch   | N       | Y             | Y          | 2x           | 44.8               |  [model](https://drive.google.com/file/d/132LEf_IDcvTMCwhAKX7x1LYWK1r34SqG/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1F87XqM3VoYOoyU7tinYirq4n7CedObDD/view?usp=sharing)|
| R-50         | pytorch   | Y       | Y             | Y          | 2x           | 47.9               |  [model](https://drive.google.com/file/d/17udTl8l3iwtqvoIOhjEi5zvPHm0HZFr4/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1EROcw3FrDP14UnWwBcswSHnzuvxLhOHz/view?usp=sharing)|
| R-101        | pytorch   | N       | Y             | N          | 2x           | 46.1               |  [model](https://drive.google.com/file/d/1uxbHTsebRnBS4Hv2ySTTwiWlulWij7bp/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1GAl1mmBRgnWa4-7jPTEdfmZGvA7PiwO4/view?usp=sharing)|

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

## Citation
```
@inproceedings{shuai2022DW,
  title={A Dual Weighting Label Assignment Scheme for Object Detection},
  author={Li, Shuai and He, Chenhang and Li, Ruihuang and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```