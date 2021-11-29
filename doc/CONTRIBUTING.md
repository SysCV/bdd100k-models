# Contributing

Thank you for your interest in contributing to our BDD100K models repository. Our goal is to provide a comprehensive list of models for each task to fascilitate research on BDD100K. You can help us by contributing your models.

#### Steps

Contributing your models is as easy as making a pull request on our repository, which anyone can do!

1. Fork and pull the latest version of BDD100K-Models.
2. Create a new branch and add your models there (see [what to add](#what-to-add)).
3. Commit your changes.
4. Create a pull request.

#### What to Add

We provide a template for each task which you should exactly follow.

- First, submit your model predictions on both the validation and test set to our evaluation server (hosted on [eval.ai](https://eval.ai/web/challenges/list)) to obtain the official results.
- Next, provide all the necessary files shown in the template.
  See the general guidelines and the task specific guidelines for the exact files to include.
- Once you have everything, submit a pull request to add your model to the README of the corresponding task and we will verify your results based on your provided information.
- If everything looks good, we will merge your PR!

For now, we are only accepting models that are/will be published in top-tier venues (e.g., CVPR, ICCV, ECCV, etc.).

#### General Guidelines

The general guidelines should be followed for any model contribution for any task.
Copy this checklist to the description of your PR and fill the box of each completed item with an X.

- [ ] Upload all your files to publicly available online storage services (e.g., [Google Drive](https://drive.google.com/)) so your files can be accessed indefinitely.
- Paper:
  - [ ] Include a link to your paper (preferably on [arXiv](https://arxiv.org/)) and the venue and year the paper is/will be published in.
  - [ ] You can add a list of authors of the paper along with links to each person's website.
  - [ ] Put the abstract of your paper in the indicated part.
- Results:
  - [ ] You can include all variations of your method (e.g., different backbones/detectors), but not baselines.
  - [ ] Include links to evaluation results on both validation and test set with BDD100K [metrics](https://doc.bdd100k.com/evaluate.html).
  - [ ] Include model weights and its corresponding MD5 hash as checksum.
  - [ ] Include model predictions and visualizations on the validation set.
- Code:
  - [ ] Include a link to your codebase on GitHub.
  - [ ] Include usage instructions so that we can easily verify your results and others can easily use your model.
  - [ ] Make sure your code and instructions are bug-free.
- [ ] Before making a pull request, make sure all the general guidelines and task specific guidelines are met.

#### License

We use the [Apache 2.0 License](../LICENSE) for our repository. The BDD100K dataset and linked repositories are not subject to the same license.

## Templates

Each task in BDD100K has its own template and guidelines. Click the links below to go to each template.

- [**Image Tagging**](#tagging)
- [**Object Detection**](#detection)
- [**Instance Segmentation**](#instance-segmentation)
- [**Semantic Segmentation**](#semantic-segmentation-and-drivable-area)
- [**Drivable Area**](#semantic-segmentation-and-drivable-area)
- [**Multiple Object Tracking (MOT)**](#mot)
- [**Multiple Object Tracking and Segmentation (MOTS)**](#mots)
- [**Pose Estimation**](#pose-estimation)

## Tagging

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Backbone | Input | Acc-val | Scores-val | Acc-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :---: | :-----: | :--------: | :------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |       |         | [scores]() |          | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the MMClassification metrics (top-1 and top-5 accuracy, precision, recall, and F1-score).
- The predictions should be a JSON file containing model predictions for the entire validation set.
- The visuals should be a zip file with tagging visualizations for the entire validation set.

Example below:

### DLA

[Deep Layer Aggregation](https://arxiv.org/abs/1707.06484) [CVPR 2018]

Authors: [Fisher Yu](https://www.yf.io/), [Dequan Wang](https://dequan.wang/), [Evan Shelhamer](http://imaginarynumber.net/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)

<details>
<summary>Abstract</summary>
Visual recognition requires rich representations that span levels from low to high, scales from small to large, and resolutions from fine to coarse. Even with the depth of features in a convolutional network, a layer in isolation is not enough: compounding and aggregating these representations improves inference of what and where. Architectural efforts are exploring many dimensions for network backbones, designing deeper or wider architectures, but how to best aggregate layers and blocks across a network deserves further attention. Although skip connections have been incorporated to combine layers, these connections have been "shallow" themselves, and only fuse by simple, one-step operations. We augment standard architectures with deeper aggregation to better fuse information across layers. Our deep layer aggregation structures iteratively and hierarchically merge the feature hierarchy to make networks with better accuracy and fewer parameters. Experiments across architectures and tasks show that deep layer aggregation improves recognition and resolution compared to existing branching and merging schemes. The code is at [this https URL](https://github.com/ucbdrive/dla).
</details>

#### Results

| Backbone |   Input    | Acc-val |                                                  Scores-val                                                   | Acc-test |                                                  Scores-test                                                   |                                Config                                |                                                                                                     Weights                                                                                                      |                                                  Preds                                                  |                                                  Visuals                                                   |
| :------: | :--------: | :-----: | :-----------------------------------------------------------------------------------------------------------: | :------: | :------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
|  DLA-34  | 224 \* 224 |  81.35  | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-val/dla34_5x_224x224_weather_tag_bdd100k.json)  |  81.24   | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-test/dla34_5x_224x224_weather_tag_bdd100k.json)  | [config](./configs/weather/dla34_5x_224x224_weather_tag_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla34_5x_224x224_weather_tag_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla34_5x_224x224_weather_tag_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/tagging/weather/preds/dla34_5x_224x224_weather_tag_bdd100k.json)  | [visuals](https://dl.cv.ethz.ch/bdd100k/tagging/weather/visuals/dla34_5x_224x224_weather_tag_bdd100k.zip)  |
|  DLA-60  | 224 \* 224 |  79.99  | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-val/dla60_5x_224x224_weather_tag_bdd100k.json)  |  79.65   | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-test/dla60_5x_224x224_weather_tag_bdd100k.json)  | [config](./configs/weather/dla60_5x_224x224_weather_tag_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla60_5x_224x224_weather_tag_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla60_5x_224x224_weather_tag_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/tagging/weather/preds/dla60_5x_224x224_weather_tag_bdd100k.json)  | [visuals](https://dl.cv.ethz.ch/bdd100k/tagging/weather/visuals/dla60_5x_224x224_weather_tag_bdd100k.zip)  |
| DLA-X-60 | 224 \* 224 |  80.22  | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-val/dla60x_5x_224x224_weather_tag_bdd100k.json) |  79.80   | [scores](https://dl.cv.ethz.ch/bdd100k/tagging/weather/scores-test/dla60x_5x_224x224_weather_tag_bdd100k.json) | [config](./configs/weather/dla60x_5x_224x224_weather_tag_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla60x_5x_224x224_weather_tag_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/dla60x_5x_224x224_weather_tag_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/tagging/weather/preds/dla60x_5x_224x224_weather_tag_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/tagging/weather/visuals/dla60x_5x_224x224_weather_tag_bdd100k.zip) |

[[Code](https://github.com/ucbdrive/dla)] [[Usage Instructions](https://github.com/SysCV/bdd100k-models/tree/main/tagging#usage)]

## Detection

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :-----: | :------: | :--------: | :--------: | :---------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |         |          |            | [scores]() |             | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K detection [metrics](https://doc.bdd100k.com/evaluate.html#detection).
- The predictions should be a JSON file containing model predictions for the entire validation set.
- The visuals should be a zip file with bounding box visualizations on the validation set.

Example below:

### Faster R-CNN

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) [NeurIPS 2015]

Authors: [Shaoqing Ren](https://www.shaoqingren.com/), [Kaiming He](http://kaiminghe.com/), [Ross Girshick](https://www.rossgirshick.info/), [Jian Sun](http://www.jiansun.org/)

<details>
<summary>Abstract</summary>
State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
</details>

#### Results

| Backbone  | Lr schd | MS-train | Box AP-val |                                           Scores-val                                            | Box AP-test |                                           Scores-test                                            |                             Config                             |                                                                                       Weights                                                                                        |                                           Preds                                           |                                           Visuals                                            |
| :-------: | :-----: | :------: | :--------: | :---------------------------------------------------------------------------------------------: | :---------: | :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| R-50-FPN  |   1x    |          |   31.04    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_1x_det_bdd100k.json)  |    29.78    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_1x_det_bdd100k.json)  | [config](./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_1x_det_bdd100k.json)  | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_1x_det_bdd100k.zip)  |
| R-50-FPN  |   3x    |    ✓     |   32.30    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_3x_det_bdd100k.json)  |    31.45    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_3x_det_bdd100k.json)  | [config](./configs/det/faster_rcnn_r50_fpn_3x_det_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_3x_det_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_3x_det_bdd100k.json)  | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_3x_det_bdd100k.zip)  |
| R-101-FPN |   3x    |    ✓     |   32.71    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r101_fpn_3x_det_bdd100k.json) |    31.96    | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/facebookresearch/detectron2)] [[Usage Instructions](https://github.com/SysCV/bdd100k-models/tree/main/det#usage)]

## Instance Segmentation

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Backbone | Lr schd | MS-train | Mask AP-val | Box AP-val | Scores-val | Mask AP-test | Box AP-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :-----: | :------: | :---------: | :--------: | :--------: | :----------: | :---------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |         |          |             |            | [scores]() |              |             | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K instance segmentation [metrics](https://doc.bdd100k.com/evaluate.html#instance-segmentation).
- The predictions should be a zip file containing model predictions for the entire validation set (both bitmasks and score JSON file).
- The visuals should be a zip file with instance segmentation visualizations on the validation set.

Example below:

### Mask R-CNN

[Mask R-CNN](https://arxiv.org/abs/1703.06870) [ICCV 2017]

Authors: [Kaiming He](http://kaiminghe.com/), [Georgia Gkioxari](https://gkioxari.github.io/), [Piotr Dollár](https://pdollar.github.io/), [Ross Girshick](https://www.rossgirshick.info/)

<details>
<summary>Abstract</summary>
We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. We hope our simple and effective approach will serve as a solid baseline and help ease future research in instance-level recognition. Code has been made available at: [this https URL](https://github.com/facebookresearch/detectron2).
</details>

#### Results

| Backbone  | Lr schd | MS-train | Mask AP-val | Box AP-val |                                              Scores-val                                               | Mask AP-test | Box AP-test |                                              Scores-test                                               |                                Config                                |                                                                                             Weights                                                                                              |                                             Preds                                              |                                              Visuals                                               |
| :-------: | :-----: | :------: | :---------: | :--------: | :---------------------------------------------------------------------------------------------------: | :----------: | :---------: | :----------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
| R-50-FPN  |   1x    |          |    16.24    |   22.34    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-val/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.json)  |    14.86     |    19.59    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-test/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.json)  | [config](./configs/ins_seg/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/ins_seg/preds/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.zip)  | [visuals](https://dl.cv.ethz.ch/bdd100k/ins_seg/visuals/mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.zip)  |
| R-50-FPN  |   3x    |    ✓     |    19.88    |   25.93    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-val/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.json)  |    17.46     |    22.32    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-test/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.json)  | [config](./configs/ins_seg/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/ins_seg/preds/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.zip)  | [visuals](https://dl.cv.ethz.ch/bdd100k/ins_seg/visuals/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.zip)  |
| R-101-FPN |   3x    |    ✓     |    20.51    |   26.08    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-val/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.json) |    17.88     |    22.01    | [scores](https://dl.cv.ethz.ch/bdd100k/ins_seg/scores-test/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.json) | [config](./configs/ins_seg/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/ins_seg/preds/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.zip) | [visuals](https://dl.cv.ethz.ch/bdd100k/ins_seg/visuals/mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.zip) |

[[Code](https://github.com/facebookresearch/detectron2)] [[Usage Instructions](https://github.com/SysCV/bdd100k-models/tree/main/ins_seg#usage)]

## Semantic Segmentation and Drivable Area

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Backbone | Iters | Input | mIoU-val | Scores-val | mIoU-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :---: | :---: | :------: | :--------: | :-------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |       |       |          | [scores]() |           | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K semantic segmentation [metrics](https://doc.bdd100k.com/evaluate.html#semantic-segmentation) or drivable area [metrics](https://doc.bdd100k.com/evaluate.html#drivable-area).
- The predictions should be a zip file containing model predictions for the entire validation set (bitmasks).
- The visuals should be a zip file with segmentation visualizations on the validation set.

Example below:

### PSPNet

[Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) [CVPR 2017]

Authors: [Hengshuang Zhao](https://hszhao.github.io/), [Jianping Shi](https://shijianping.me/), [Xiaojuan Qi](https://xjqi.github.io/), [Xiaogang Wang](https://www.ee.cuhk.edu.hk/~xgwang/), [Jiaya Jia](https://jiaya.me/)

<details>
<summary>Abstract</summary>
Scene parsing is challenging for unrestricted open vocabulary and diverse scenes. In this paper, we exploit the capability of global context information by different-region-based context aggregation through our pyramid pooling module together with the proposed pyramid scene parsing network (PSPNet). Our global prior representation is effective to produce good quality results on the scene parsing task, while PSPNet provides a superior framework for pixel-level prediction tasks. The proposed approach achieves state-of-the-art performance on various datasets. It came first in ImageNet scene parsing challenge 2016, PASCAL VOC 2012 benchmark and Cityscapes benchmark. A single PSPNet yields new record of mIoU accuracy 85.4\% on PASCAL VOC 2012 and accuracy 80.2\% on Cityscapes.
</details>

#### Results

| Backbone | Iters |    Input    | mIoU-val |                                                 Scores-val                                                  | mIoU-test |                                                 Scores-test                                                  |                                   Config                                   |                                                                                                   Weights                                                                                                    |                                                Preds                                                 |                                                 Visuals                                                  |
| :------: | :---: | :---------: | :------: | :---------------------------------------------------------------------------------------------------------: | :-------: | :----------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| R-50-D8  |  40K  | 512 \* 1024 |  61.88   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-val/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.json)  |   54.50   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-test/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.json)  | [config](./configs/sem_seg/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/sem_seg/preds/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.zip)  | [visuals](https://dl.cv.ethz.ch/bdd100k/sem_seg/visuals/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.zip)  |
| R-50-D8  |  80K  | 512 \* 1024 |  62.03   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-val/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.json)  |   54.99   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-test/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.json)  | [config](./configs/sem_seg/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.py)  |  [model](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.md5)  | [preds](https://dl.cv.ethz.ch/bdd100k/sem_seg/preds/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.zip)  | [visuals](https://dl.cv.ethz.ch/bdd100k/sem_seg/visuals/pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.zip)  |
| R-101-D8 |  80K  | 512 \* 1024 |  63.62   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-val/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.json) |   56.32   | [scores](https://dl.cv.ethz.ch/bdd100k/sem_seg/scores-test/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.json) | [config](./configs/sem_seg/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/sem_seg/preds/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.zip) | [visuals](https://dl.cv.ethz.ch/bdd100k/sem_seg/visuals/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.zip) |

[[Code](https://github.com/hszhao/PSPNet)] [[Usage Instructions](https://github.com/SysCV/bdd100k-models/tree/main/sem_seg#usage)]

## MOT

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Detector | mMOTA-val | mIDF1-val | ID Sw.-val | Scores-val | mMOTA-test | mIDF1-test | ID Sw.-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :-------: | :-------: | :--------: | :--------: | :--------: | :--------: | :---------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |           |           |            | [scores]() |            |            |             | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K MOT [metrics](https://doc.bdd100k.com/evaluate.html#multiple-object-tracking).
- The predictions should be a JSON file containing model predictions for the entire validation set.
- The visuals should be a zip file with bounding box tracking visualizations on the validation set. Can be images or videos.

Example below:

### QDTrack

[Quasi-Dense Similarity Learning for Multiple Object Tracking](https://arxiv.org/abs/2006.06664) [CVPR 2021 Oral]

Authors: [Jiangmiao Pang](https://scholar.google.com/citations?user=ssSfKpAAAAAJ), Linlu Qiu, [Xia Li](https://xialipku.github.io/), [Haofeng Chen](https://www.haofeng.io/), Qi Li, [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Fisher Yu](https://www.yf.io/)

<details>
<summary>Abstract</summary>
Similarity learning has been recognized as a crucial step for object tracking. However, existing multiple object tracking methods only use sparse ground truth matching as the training objective, while ignoring the majority of the informative regions on the images. In this paper, we present Quasi-Dense Similarity Learning, which densely samples hundreds of region proposals on a pair of images for contrastive learning. We can naturally combine this similarity learning with existing detection methods to build Quasi-Dense Tracking (QDTrack) without turning to displacement regression or motion priors. We also find that the resulting distinctive feature space admits a simple nearest neighbor search at the inference time. Despite its simplicity, QDTrack outperforms all existing methods on MOT, BDD100K, Waymo, and TAO tracking benchmarks. It achieves 68.7 MOTA at 20.3 FPS on MOT17 without using external training data. Compared to methods with similar detectors, it boosts almost 10 points of MOTA and significantly decreases the number of ID switches on BDD100K and Waymo datasets.
</details>

#### Results

| Detector  | mMOTA-val | mIDF1-val | ID Sw.-val |                                            Scores-val                                             | mMOTA-test | mIDF1-test | ID Sw.-test |                                            Scores-test                                             |                                                   Config                                                    |                                                                                         Weights                                                                                          |                                            Preds                                            |                                            Visuals                                             |
| :-------: | :-------: | :-------: | :--------: | :-----------------------------------------------------------------------------------------------: | :--------: | :--------: | :---------: | :------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: |
| ResNet-50 |   36.6    |   51.6    |    6193    | [scores](https://dl.cv.ethz.ch/bdd100k/mot/scores-val/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.json) |    35.7    |    52.3    |    10822    | [scores](https://dl.cv.ethz.ch/bdd100k/mot/scores-test/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.json) | [config](https://github.com/SysCV/qdtrack/blob/master/configs/bdd100k/qdtrack-frcnn_r50_fpn_12e_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/mot/models/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/mot/models/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/mot/preds/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/mot/visuals/qdtrack-frcnn_r50_fpn_12e_mot_bdd100k.zip) |

[[Code](https://github.com/SysCV/qdtrack)] [[Usage Instructions](https://github.com/SysCV/qdtrack/blob/master/docs/GET_STARTED.md)]

## MOTS

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Detector | mMOTSA-val | mIDF1-val | ID Sw.-val | Scores-val | mMOTSA-test | mIDF1-test | ID Sw.-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :--------: | :-------: | :--------: | :--------: | :---------: | :--------: | :---------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
|          |            |           |            | [scores]() |             |            |             | [scores]()  | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K MOTS [metrics](https://doc.bdd100k.com/evaluate.html#multi-object-tracking-and-segmentation-segmentation-tracking).
- The predictions should be a zip file containing model predictions for the entire validation set (bitmasks).
- The visuals should be a zip file with segmentation tracking visualizations on the validation set. Can be images or videos.

Example below:

### PCAN

[Prototypical Cross-Attention Networks (PCAN) for Multiple Object Tracking and Segmentation](https://arxiv.org/abs/2106.11958) [NeurIPS 2021 Spotlight]

Authors: [Lei Ke](https://www.kelei.site/), [Xia Li](https://xialipku.github.io/), [Martin Danelljan](https://martin-danelljan.github.io/), [Yu-Wing Tai](https://cse.hkust.edu.hk/admin/people/faculty/profile/yuwing?u=yuwing), [Chi-Keung Tang](https://cse.hkust.edu.hk/admin/people/faculty/profile/cktang?u=cktang), [Fisher Yu](https://www.yf.io/)

<details>
<summary>Abstract</summary>
Multiple object tracking and segmentation requires detecting, tracking, and segmenting objects belonging to a set of given classes. Most approaches only exploit the temporal dimension to address the association problem, while relying on single frame predictions for the segmentation mask itself. We propose Prototypical Cross-Attention Network (PCAN), capable of leveraging rich spatio-temporal information for online multiple object tracking and segmentation. PCAN first distills a space-time memory into a set of prototypes and then employs cross-attention to retrieve rich information from the past frames. To segment each object, PCAN adopts a prototypical appearance module to learn a set of contrastive foreground and background prototypes, which are then propagated over time. Extensive experiments demonstrate that PCAN outperforms current video instance tracking and segmentation competition winners on both Youtube-VIS and BDD100K datasets, and shows efficacy to both one-stage and two-stage segmentation frameworks.
</details>

#### Results

| Detector  | mMOTSA-val | mIDF1-val | ID Sw.-val |                                            Scores-val                                            | mMOTSA-test | mIDF1-test | ID Sw.-test |                                            Scores-test                                            |                                             Config                                             |                                                                                        Weights                                                                                         |                                           Preds                                           |                                            Visuals                                            |
| :-------: | :--------: | :-------: | :--------: | :----------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :-----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| ResNet-50 |    28.1    |   45.4    |    874     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-val/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) |    31.9     |    50.4    |     845     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-test/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) | [config](https://github.com/SysCV/pcan/blob/main/configs/segtrack-frcnn_r50_fpn_12e_bdd10k.py) | [model](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/mots/preds/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) | [visuals](https://dl.cv.ethz.ch/bdd100k/mots/visuals/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) |

[[Code](https://github.com/SysCV/pcan)] [[Usage Instructions](https://github.com/SysCV/pcan/blob/main/docs/GET_STARTED.md)]

## Pose Estimation

Template and guidelines below:

### Method Name

[Paper name]() [Venue and Year]

Authors: Author list

<details>
<summary>Abstract</summary>
Put your abstract here.
</details>

#### Results

| Backbone | Input Size | Pose AP-val | Scores-val | Pose AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|          |            |           | [scores]() |             | [scores]() | [config]() | [model]() \| [MD5]() | [preds]() | [visuals]() |

[[Code]()] [[Usage Instructions]()]

Other information.

### Guidelines

- The scores file should be a JSON file with evaluation results for all the BDD100K pose estimation [metrics](https://doc.bdd100k.com/evaluate.html#pose-estimation).
- The predictions should be a JSON file containing model predictions for the entire validation set.
- The visuals should be a zip file with pose visualizations on the validation set.

Example below:

### HRNet

[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919) [CVPR 2019 / TPAMI 2020]

Authors: [Jingdong Wang](https://jingdongwang2017.github.io/), [Ke Sun](https://github.com/sunke123), [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ), Borui Jiang, Chaorui Deng, [Yang Zhao](https://yangyangkiki.github.io/), Dong Liu, [Yadong Mu](http://www.muyadong.com/), Mingkui Tan, [Xinggang Wang](https://xinggangw.info/), [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/), [Bin Xiao](https://www.microsoft.com/en-us/research/people/bixi/)

<details>
<summary>Abstract</summary>
High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at [this https URL](https://github.com/HRNet).
</details>

#### Results

| Backbone | Input Size | Pose AP-val | Scores-val | Pose AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| HRNet-w32 | 256 * 192 | 48.83 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-val/hrnet_w32_256x192_pose_bdd100k.json) | 46.13 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-test/hrnet_w32_256x192_pose_bdd100k.json) | [config](./configs/hrnet_w32_256x192_pose_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w32_256x192_pose_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w32_256x192_pose_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/pose/preds/hrnet_w32_256x192_pose_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/pose/visuals/hrnet_w32_256x192_pose_bdd100k.zip) |
| HRNet-w48 | 256 * 192 | 50.32 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-val/hrnet_w48_256x192_pose_bdd100k.json) | 47.36 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-test/hrnet_w48_256x192_pose_bdd100k.json) | [config](./configs/hrnet_w48_256x192_pose_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w48_256x192_pose_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w48_256x192_pose_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/pose/preds/hrnet_w48_256x192_pose_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/pose/visuals/hrnet_w48_256x192_pose_bdd100k.zip) |
| HRNet-w32 | 320 * 256 | 49.86 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-val/hrnet_w32_320x256_pose_bdd100k.json) | 46.90 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-test/hrnet_w32_320x256_pose_bdd100k.json) | [config](./configs/hrnet_w32_320x256_pose_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w32_320x256_pose_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w32_320x256_pose_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/pose/preds/hrnet_w32_320x256_pose_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/pose/visuals/hrnet_w32_320x256_pose_bdd100k.zip) |
| HRNet-w48 | 320 * 256 | 50.16 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-val/hrnet_w48_320x256_pose_bdd100k.json) | 47.32 | [scores](https://dl.cv.ethz.ch/bdd100k/pose/scores-test/hrnet_w48_320x256_pose_bdd100k.json) | [config](./configs/hrnet_w48_320x256_pose_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w48_320x256_pose_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w48_320x256_pose_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/pose/preds/hrnet_w48_320x256_pose_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/pose/visuals/hrnet_w48_320x256_pose_bdd100k.zip) |

[[Code](https://github.com/HRNet)] [[Usage Instructions](https://github.com/SysCV/bdd100k-models/tree/main/pose#usage)]
