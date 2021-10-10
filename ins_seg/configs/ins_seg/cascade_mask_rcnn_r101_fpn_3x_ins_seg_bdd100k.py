"""Cascade Mask RCNN with ResNet101-FPN, 3x schedule, MS training."""

_base_ = "./cascade_mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/cascade_mask_rcnn_r101_fpn_3x_ins_seg_bdd100k.pth"
