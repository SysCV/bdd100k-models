"""Sparse RCNN with ResNet101-FPN, 100 proposals, 3x schedule, MS training."""

_base_ = "./sparse_rcnn_r50_fpn_3x_det_bdd100k.py"

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r101_fpn_3x_det_bdd100k.pth"
