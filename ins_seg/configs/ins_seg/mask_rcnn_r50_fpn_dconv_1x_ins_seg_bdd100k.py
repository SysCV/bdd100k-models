"""Deformable Conv Nets with ResNet50-FPN, 1x schedule."""

_base_ = "./mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.py"
model = dict(
    backbone=dict(
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_dconv_1x_ins_seg_bdd100k.pth"
