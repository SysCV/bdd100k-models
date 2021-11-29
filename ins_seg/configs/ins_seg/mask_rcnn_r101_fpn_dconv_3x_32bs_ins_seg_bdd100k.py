"""Deformable Conv Nets with ResNet101-FPN, 3x schedule, MS training, 32 batch size."""

_base_ = "./mask_rcnn_r101_fpn_3x_32bs_ins_seg_bdd100k.py"
model = dict(
    backbone=dict(
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r101_fpn_dconv_3x_32bs_ins_seg_bdd100k.pth"
