"""ATSS with ResNet101-FPN and DyHead, 3x schedule, MS training."""

_base_ = "./atss_r50_fpn_dyhead_3x_det_bdd100k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/atss_r101_fpn_dyhead_3x_det_bdd100k.pth"
