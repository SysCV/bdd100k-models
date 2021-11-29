# Detection Models of BDD100K

The object detection task involves localization (predicting a bounding box for each object) and classification (predicting the object category).

![det1](../doc/images/det1.png)

The BDD100K dataset contains bounding box annotations for 100K images (70K/10K/20K for train/val/test). Each annotation contains bounding box labels for 10 object classes. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

## Model Zoo

For training the models listed below, we follow the common settings used by MMDetection (details [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md#common-settings)), unless otherwise stated.
All models are trained on either 8 GeForce RTX 2080 Ti GPUs or 8 TITAN RTX GPUs with a batch size of 4x8=32.
See the config files for the detailed setting for each model.

---

### Faster R-CNN

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) [NeurIPS 2015]

Authors: [Shaoqing Ren](https://www.shaoqingren.com/), [Kaiming He](http://kaiminghe.com/), [Ross Girshick](https://www.rossgirshick.info/), [Jian Sun](http://www.jiansun.org/)

<details>
<summary>Abstract</summary>
State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection networks, exposing region proposal computation as a bottleneck. In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. The RPN is trained end-to-end to generate high-quality region proposals, which are used by Fast R-CNN for detection. We further merge RPN and Fast R-CNN into a single network by sharing their convolutional features---using the recently popular terminology of neural networks with 'attention' mechanisms, the RPN component tells the unified network where to look. For the very deep VGG-16 model, our detection system has a frame rate of 5fps (including all steps) on a GPU, while achieving state-of-the-art object detection accuracy on PASCAL VOC 2007, 2012, and MS COCO datasets with only 300 proposals per image. In ILSVRC and COCO 2015 competitions, Faster R-CNN and RPN are the foundations of the 1st-place winning entries in several tracks. Code has been made publicly available.
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 31.04 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_1x_det_bdd100k.json) | 29.78 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 32.30 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_3x_det_bdd100k.json) | 31.45 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 32.71 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r101_fpn_3x_det_bdd100k.json) | 31.96 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/facebookresearch/detectron2)] [[Usage Instructions](#usage)]

---

### RetinaNet

[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) [ICCV 2017]

Authors: [Tsung-Yi Lin](https://scholar.google.com/citations?user=_BPdgV0AAAAJ), [Priya Goyal](https://research.fb.com/people/goyal-priya/), [Ross Girshick](https://www.rossgirshick.info/), [Kaiming He](http://kaiminghe.com/), [Piotr Dollár](https://pdollar.github.io/)

<details>
<summary>Abstract</summary>
The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stage detectors that are applied over a regular, dense sampling of possible object locations have the potential to be faster and simpler, but have trailed the accuracy of two-stage detectors thus far. In this paper, we investigate why this is the case. We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training. To evaluate the effectiveness of our loss, we design and train a simple dense detector we call RetinaNet. Our results show that when trained with the focal loss, RetinaNet is able to match the speed of previous one-stage detectors while surpassing the accuracy of all existing state-of-the-art two-stage detectors. Code is at: [this https URL](https://github.com/facebookresearch/detectron2).
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 28.58 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/retinanet_r50_fpn_1x_det_bdd100k.json) | 27.14 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/retinanet_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/retinanet_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/retinanet_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/retinanet_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 30.91 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/retinanet_r50_fpn_3x_det_bdd100k.json) | 30.21 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/retinanet_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/retinanet_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/retinanet_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/retinanet_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 31.29 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/retinanet_r101_fpn_3x_det_bdd100k.json) | 30.62 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/retinanet_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/retinanet_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/retinanet_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/retinanet_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/facebookresearch/detectron2)] [[Usage Instructions](#usage)]

---

### Cascade R-CNN

[Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726) [CVPR 2018]

Authors: [Zhaowei Cai](https://zhaoweicai.github.io/), [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/)

<details>
<summary>Abstract</summary>
In object detection, an intersection over union (IoU) threshold is required to define positives and negatives. An object detector, trained with low IoU threshold, e.g. 0.5, usually produces noisy detections. However, detection performance tends to degrade with increasing the IoU thresholds. Two main factors are responsible for this: 1) overfitting during training, due to exponentially vanishing positive samples, and 2) inference-time mismatch between the IoUs for which the detector is optimal and those of the input hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, is proposed to address these problems. It consists of a sequence of detectors trained with increasing IoU thresholds, to be sequentially more selective against close false positives. The detectors are trained stage by stage, leveraging the observation that the output of a detector is a good distribution for training the next higher quality detector. The resampling of progressively improved hypotheses guarantees that all detectors have a positive set of examples of equivalent size, reducing the overfitting problem. The same cascade procedure is applied at inference, enabling a closer match between the hypotheses and the detector quality of each stage. A simple implementation of the Cascade R-CNN is shown to surpass all single-model object detectors on the challenging COCO dataset. Experiments also show that the Cascade R-CNN is widely applicable across detector architectures, achieving consistent gains independently of the baseline detector strength. The code will be made available at this [https URL](https://github.com/zhaoweicai/cascade-rcnn).
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 32.40 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/cascade_rcnn_r50_fpn_1x_det_bdd100k.json) | 31.23 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/cascade_rcnn_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/cascade_rcnn_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/cascade_rcnn_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/cascade_rcnn_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 33.72 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/cascade_rcnn_r50_fpn_3x_det_bdd100k.json) | 33.07 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/cascade_rcnn_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/cascade_rcnn_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/cascade_rcnn_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/cascade_rcnn_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 33.57 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/cascade_rcnn_r101_fpn_3x_det_bdd100k.json) | 32.90 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/cascade_rcnn_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/cascade_rcnn_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/cascade_rcnn_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/cascade_rcnn_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/zhaoweicai/cascade-rcnn)] [[Usage Instructions](#usage)]

---

### FCOS

[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) [ICCV 2019]

Authors: [Zhi Tian](https://zhitian.xyz/), [Chunhua Shen](https://cshen.github.io/), [Hao Chen](https://stan-haochen.github.io/), [Tong He](https://tonghehehe.com/)

<details>
<summary>Abstract</summary>
We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor box free, as well as proposal free. By eliminating the predefined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), FCOS with ResNeXt-64x4d-101 achieves 44.7\% in AP with single-model and single-scale testing, surpassing previous one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks. Code is available at:Code is available at: [this https URL](https://github.com/tianzhi0549/FCOS/).
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 27.69 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/fcos_r50_fpn_1x_det_bdd100k.json) | 26.16 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/fcos_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/fcos_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/fcos_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/fcos_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 30.60 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/fcos_r50_fpn_3x_det_bdd100k.json) | 28.96 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/fcos_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/fcos_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/fcos_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/fcos_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 31.13 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/fcos_r101_fpn_3x_det_bdd100k.json) | 29.62 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/fcos_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/fcos_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/fcos_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/fcos_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/tianzhi0549/FCOS/)] [[Usage Instructions](#usage)]

---

### Deformable ConvNets v2

[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168) [CVPR 2019]

Authors: [Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ), [Han Hu](https://sites.google.com/site/hanhushomepage/), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), [Jifeng Dai](https://jifengdai.org/)

<details>
<summary>Abstract</summary>
The superior performance of Deformable Convolutional Networks arises from its ability to adapt to the geometric variations of objects. Through an examination of its adaptive behavior, we observe that while the spatial support for its neural features conforms more closely than regular ConvNets to object structure, this support may nevertheless extend well beyond the region of interest, causing features to be influenced by irrelevant image content. To address this problem, we present a reformulation of Deformable ConvNets that improves its ability to focus on pertinent image regions, through increased modeling power and stronger training. The modeling power is enhanced through a more comprehensive integration of deformable convolution within the network, and by introducing a modulation mechanism that expands the scope of deformation modeling. To effectively harness this enriched modeling capability, we guide network training via a proposed feature mimicking scheme that helps the network to learn features that reflect the object focus and classification power of R-CNN features. With the proposed contributions, this new version of Deformable ConvNets yields significant performance gains over the original model and produces leading results on the COCO benchmark for object detection and instance segmentation.
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 32.09 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.json) | 30.93 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_deconv_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 33.21 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.json) | 32.41 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r50_fpn_deconv_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 33.09 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.json) | 32.43 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_r101_fpn_deconv_3x_det_bdd100k.zip) |

[[Code](https://github.com/msracver/Deformable-ConvNets)] [[Usage Instructions](#usage)]

---

### Libra R-CNN

[Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/abs/1904.02701) [CVPR 2019]

Authors: [Jiangmiao Pang](https://scholar.google.com/citations?user=ssSfKpAAAAAJ), [Kai Chen](https://chenkai.site/), [Jianping Shi](https://shijianping.me/), Huajun Feng, [Wanli Ouyang](https://wlouyang.github.io/), [Dahua Lin](http://dahua.site/)

<details>
<summary>Abstract</summary>
Compared with model architectures, the training process, which is also crucial to the success of detectors, has received relatively less attention in object detection. In this work, we carefully revisit the standard training practice of detectors, and find that the detection performance is often limited by the imbalance during the training process, which generally consists in three levels - sample level, feature level, and objective level. To mitigate the adverse effects caused thereby, we propose Libra R-CNN, a simple but effective framework towards balanced learning for object detection. It integrates three novel components: IoU-balanced sampling, balanced feature pyramid, and balanced L1 loss, respectively for reducing the imbalance at sample, feature, and objective level. Benefitted from the overall balanced design, Libra R-CNN significantly improves the detection performance. Without bells and whistles, it achieves 2.5 points and 2.0 points higher Average Precision (AP) than FPN Faster R-CNN and RetinaNet respectively on MSCOCO.
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 30.70 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/libra_faster_r50_fpn_1x_det_bdd100k.json) | 29.54 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/libra_faster_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/libra_faster_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/libra_faster_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/libra_faster_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 32.00 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/libra_faster_r50_fpn_3x_det_bdd100k.json) | 31.05 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/libra_faster_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/libra_faster_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/libra_faster_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/libra_faster_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 32.24 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/libra_faster_r101_fpn_3x_det_bdd100k.json) | 31.49 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/libra_faster_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/libra_faster_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/libra_faster_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/libra_faster_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/libra_faster_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/OceanPang/Libra_R-CNN)] [[Usage Instructions](#usage)]

---

### HRNet

[Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919) [CVPR 2019 / TPAMI 2020]

Authors: [Jingdong Wang](https://jingdongwang2017.github.io/), [Ke Sun](https://github.com/sunke123), [Tianheng Cheng](https://scholar.google.com/citations?user=PH8rJHYAAAAJ), Borui Jiang, Chaorui Deng, [Yang Zhao](https://yangyangkiki.github.io/), Dong Liu, [Yadong Mu](http://www.muyadong.com/), Mingkui Tan, [Xinggang Wang](https://xinggangw.info/), [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/), [Bin Xiao](https://www.microsoft.com/en-us/research/people/bixi/)

<details>
<summary>Abstract</summary>
High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. We show the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems. All the codes are available at [this https URL](https://github.com/HRNet).
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| HRNet-w18 | 1x |  | 31.74 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.json) | 30.64 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_hrnetv2p_w18_1x_det_bdd100k.zip) |
| HRNet-w18 | 3x | ✓ | 33.35 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.json) | 32.61 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.zip) |
| HRNet-w32 | 1x |  | 32.84 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.json) | 31.84 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_hrnetv2p_w32_1x_det_bdd100k.zip) |
| HRNet-w32 | 3x | ✓ | 33.97 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.json) | 33.19 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.json) | [config](./configs/det/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.zip) |

[[Code](https://github.com/HRNet)] [[Usage Instructions](#usage)]

---

### Sparse R-CNN

[Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450) [CVPR 2021]

Authors: [Peize Sun](https://peizesun.github.io/), Rufeng Zhang, Yi Jiang, [Tao Kong](http://www.taokong.org/), [Chenfeng Xu](https://scholar.google.com/citations?user=RpqvaTUAAAAJ), [Wei Zhan](https://zhanwei.site/), [Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/), [Lei Li](https://sites.cs.ucsb.edu/~lilei/), [Zehuan Yuan](https://shallowyuan.github.io/), [Changhu Wang](https://changhu.wang/), [Ping Luo](http://luoping.me/)

<details>
<summary>Abstract</summary>
We present Sparse R-CNN, a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as k anchor boxes pre-defined on all grids of image feature map of size H×W. In our method, however, a fixed sparse set of learned object proposals, total length of N, are provided to object recognition head to perform classification and location. By eliminating HWk (up to hundreds of thousands) hand-designed object candidates to N (e.g. 100) learnable proposals, Sparse R-CNN completely avoids all efforts related to object candidates design and many-to-one label assignment. More importantly, final predictions are directly output without non-maximum suppression post-procedure. Sparse R-CNN demonstrates accuracy, run-time and training convergence performance on par with the well-established detector baselines on the challenging COCO dataset, e.g., achieving 45.0 AP in standard 3× training schedule and running at 22 fps using ResNet-50 FPN model. We hope our work could inspire re-thinking the convention of dense prior in object detectors. The code is available at: [this https URL](https://github.com/PeizeSun/SparseR-CNN).
</details>

#### Results

| Backbone | Lr schd | MS-train | Box AP-val | Scores-val | Box AP-test | Scores-test | Config | Weights | Preds | Visuals |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| R-50-FPN | 1x |  | 26.71 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/sparse_rcnn_r50_fpn_1x_det_bdd100k.json) | 25.55 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/sparse_rcnn_r50_fpn_1x_det_bdd100k.json) | [config](./configs/det/sparse_rcnn_r50_fpn_1x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r50_fpn_1x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r50_fpn_1x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/sparse_rcnn_r50_fpn_1x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/sparse_rcnn_r50_fpn_1x_det_bdd100k.zip) |
| R-50-FPN | 3x | ✓ | 31.31 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/sparse_rcnn_r50_fpn_3x_det_bdd100k.json) | 31.19 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/sparse_rcnn_r50_fpn_3x_det_bdd100k.json) | [config](./configs/det/sparse_rcnn_r50_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r50_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r50_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/sparse_rcnn_r50_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/sparse_rcnn_r50_fpn_3x_det_bdd100k.zip) |
| R-101-FPN | 3x | ✓ | 32.18 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-val/sparse_rcnn_r101_fpn_3x_det_bdd100k.json) | 31.45 | [scores](https://dl.cv.ethz.ch/bdd100k/det/scores-test/sparse_rcnn_r101_fpn_3x_det_bdd100k.json) | [config](./configs/det/sparse_rcnn_r101_fpn_3x_det_bdd100k.py) | [model](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r101_fpn_3x_det_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r101_fpn_3x_det_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/det/preds/sparse_rcnn_r101_fpn_3x_det_bdd100k.json) | [visuals](https://dl.cv.ethz.ch/bdd100k/det/visuals/sparse_rcnn_r101_fpn_3x_det_bdd100k.zip) |

[[Code](https://github.com/PeizeSun/SparseR-CNN)] [[Usage Instructions](#usage)]

---

## Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n bdd100k-mmdet python=3.8
conda activate bdd100k-mmdet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install mmcv and mmdetection.

```shell
pip install mmcv-full
pip install mmdet
```

You can also refer to the [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).

Note that mmdetection uses their forked version of pycocotools via the github repo instead of pypi for better compatibility. If you meet issues, you may need to re-install the cocoapi through

```shell
pip uninstall pycocotools
pip install git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools
```

## Usage

### Model Inference

Single GPU inference:

```shell
python ./test.py ${CONFIG_FILE} --format-only --format-dir ${OUTPUT_DIR} [--cfg-options]
```

Multiple GPU inference:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=12000 ./test.py $CFG_FILE \
    --format-only --format-dir ${OUTPUT_DIR} [--cfg-options] \
    --launcher pytorch
```

### Output Evaluation

#### Validation Set

To evaluate the detection performance on the BDD100K validation set, you can follow the official evaluation [scripts](https://doc.bdd100k.com/evaluate.html) provided by BDD100K:

```bash
python -m bdd100k.eval.run -t det \
    -g ../data/bdd100k/labels/det_20/det_${SET_NAME}.json \
    -r ${OUTPUT_DIR}/det.json [--out-file ${RESULTS_FILE}] [--nproc ${NUM_PROCESS}]
```

#### Test Set

You can obtain the performance on the BDD100K test set by submitting your model predictions to our [evaluation server](https://eval.ai/web/challenges/challenge-page/1260) hosted on EvalAI.

### Output Visualization

For visualization, you can use the visualization tool provided by [Scalabel](https://doc.scalabel.ai/visual.html).

Below is an example:

```python
import os
import numpy as np
from PIL import Image
from scalabel.label.io import load
from scalabel.vis.label import LabelViewer

# load prediction frames
frames = load('$OUTPUT_DIR/det.json').frames

viewer = LabelViewer()
for frame in frames:
    img = np.array(Image.open(os.path.join('$IMG_DIR', frame.name)))
    viewer.draw(img, frame)
    viewer.savefig(os.path.join('$VIS_DIR', frame.name))
```

## Contribution

**You can include your models in this repo as well!** Please follow the [contribution](../doc/CONTRIBUTING.md) instructions.
