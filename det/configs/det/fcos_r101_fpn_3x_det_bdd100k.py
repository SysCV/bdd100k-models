"""FCOS with ResNet101-FPN, 3x schedule, MS training."""

_base_ = "./fcos_r50_fpn_3x_det_bdd100k.py"
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="open-mmlab://detectron/resnet101_caffe",
        ),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/fcos_r101_fpn_3x_det_bdd100k.pth"
