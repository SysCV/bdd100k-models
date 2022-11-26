"""Faster RCNN with ResNet101-FPN, 5x schedule, MS training."""

_base_ = "./faster_rcnn_r50_fpn_5x_det_bdd100k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r101_fpn_5x_det_bdd100k.pth"
