# Multiple Object Tracking and Segmentation Models of BDD100K

The multiple object tracking and segmentation (MOTS) task involves detecting, tracking, and segmenting objects of interest throughout each video sequence.

![seg_track1](../doc/images/seg_track1.gif)

The BDD100K dataset contains MOTS annotations for 223 videos (154/32/37 for train/val/test) with 8 categories. Each video is approximately 40 seconds and annotated at 5 fps, resulting in around 200 frames per video. For details about downloading the data and the annotation format for this task, see the [official documentation](https://doc.bdd100k.com/download.html).

## Model Zoo

### PCAN

[Prototypical Cross-Attention Networks (PCAN) for Multiple Object Tracking and Segmentation](https://arxiv.org/abs/2106.11958) [NeurIPS 2021 Spotlight]

Authors: [Lei Ke](https://www.kelei.site/), [Xia Li](https://xialipku.github.io/), [Martin Danelljan](https://martin-danelljan.github.io/), [Yu-Wing Tai](https://cse.hkust.edu.hk/admin/people/faculty/profile/yuwing?u=yuwing), [Chi-Keung Tang](https://cse.hkust.edu.hk/admin/people/faculty/profile/cktang?u=cktang), [Fisher Yu](https://www.yf.io/)

<details>
<summary>Abstract</summary>
Multiple object tracking and segmentation requires detecting, tracking, and segmenting objects belonging to a set of given classes. Most approaches only exploit the temporal dimension to address the association problem, while relying on single frame predictions for the segmentation mask itself. We propose Prototypical Cross-Attention Network (PCAN), capable of leveraging rich spatio-temporal information for online multiple object tracking and segmentation. PCAN first distills a space-time memory into a set of prototypes and then employs cross-attention to retrieve rich information from the past frames. To segment each object, PCAN adopts a prototypical appearance module to learn a set of contrastive foreground and background prototypes, which are then propagated over time. Extensive experiments demonstrate that PCAN outperforms current video instance tracking and segmentation competition winners on both Youtube-VIS and BDD100K datasets, and shows efficacy to both one-stage and two-stage segmentation frameworks.
</details>

#### Results

| Detector | mMOTSA-val | mIDF1-val | ID Sw.-val | Scores-val | mMOTSA-test | mIDF1-test | ID Sw.-test | Scores-test |   Config   |       Weights        |   Preds   |   Visuals   |
| :------: | :--------: | :-------: | :--------: | :--------: | :---------: | :--------: | :---------: | :---------: | :--------: | :------------------: | :-------: | :---------: |
| ResNet-50 |    28.1    |   45.4    |    874     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-val/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) |    31.9     |    50.4    |     845     | [scores](https://dl.cv.ethz.ch/bdd100k/mots/scores-test/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) | [config](https://github.com/SysCV/pcan/blob/main/configs/segtrack-frcnn_r50_fpn_12e_bdd10k.py) | [model](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.pth) \| [MD5](https://dl.cv.ethz.ch/bdd100k/mots/models/pcan-frcnn_r50_fpn_12e_mots_bdd100k.md5) | [preds](https://dl.cv.ethz.ch/bdd100k/mots/preds/pcan-frcnn_r50_fpn_12e_mots_bdd100k.json) \| [masks](https://dl.cv.ethz.ch/bdd100k/mots/preds/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) | [visuals](https://dl.cv.ethz.ch/bdd100k/mots/visuals/pcan-frcnn_r50_fpn_12e_mots_bdd100k.zip) |

[[Code](https://github.com/SysCV/pcan)] [[Usage Instructions](https://github.com/SysCV/pcan/blob/main/docs/GET_STARTED.md)]

## Usage

### Model Inference

For model inference, please refer to the usage instructions of the corresponding model.

### Output Evaluation

#### Validation Set

To evaluate the MOT performance on the BDD100K validation set, you can follow the official evaluation [scripts](https://doc.bdd100k.com/evaluate.html) provided by BDD100K:

```bash
python -m bdd100k.eval.run -t seg_track \
    -g ../data/bdd100k/labels/seg_track_20/${SET_NAME} \
    -r ${OUTPUT_DIR} \
    [--out-file ${RESULTS_FILE}] [--nproc ${NUM_PROCESS}]
```

#### Test Set

You can obtain the performance on the BDD100K test set by submitting your model predictions to our [evaluation server](https://eval.ai/web/challenges/challenge-page/1295) hosted on EvalAI.

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
