# Multiple Object Tracking Models of BDD100K

The multiple object tracking (MOT) task involves detecting and tracking objects of interest throughout each video sequence.

![box_track1](../doc/images/box_track1.gif)

The BDD100K dataset contains MOT annotations for 2K videos (1.4K/200/400 for train/val/test) with 8 categories. Each video is approximately 40 seconds and annotated at 5 fps, resulting in around 200 frames per video. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

## Model Zoo

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

## Usage

### Model Inference

For model inference, please refer to the usage instructions of the corresponding model.

### Output Evaluation

#### Validation Set

To evaluate the MOT performance on the BDD100K validation set, you can follow the official evaluation [scripts](https://doc.bdd100k.com/evaluate.html) provided by BDD100K:

```bash
python -m bdd100k.eval.run -t box_track \
    -g ../data/bdd100k/labels/box_track_20/${SET_NAME} \
    -r ${OUTPUT_FILE} \
    [--out-file ${RESULTS_FILE}] [--nproc ${NUM_PROCESS}]
```

#### Test Set

You can obtain the performance on the BDD100K test set by submitting your model predictions to our [evaluation server](https://eval.ai/web/challenges/challenge-page/1259) hosted on EvalAI.

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
frames = load('$OUTPUT_FILE').frames

viewer = LabelViewer()
for frame in frames:
    img = np.array(Image.open(os.path.join('$IMG_DIR', frame.name)))
    viewer.draw(img, frame)
    viewer.save(os.path.join('$VIS_DIR', frame.videoName, frame.name))
```

## Contribution

**You can include your models in this repo as well!** Please follow the [contribution](../doc/CONTRIBUTING.md) instructions.
