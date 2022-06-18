"""Panoptic FPN with ResNet101-FPN, 5x schedule, MS training."""

_base_ = "./panoptic_fpn_r50_fpn_5x_pan_seg_bdd100k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)

custom_hooks = []
load_from = "https://dl.cv.ethz.ch/bdd100k/pan_seg/models/panoptic_fpn_r101_fpn_5x_pan_seg_bdd100k.pth"
