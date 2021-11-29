"""HRNet40, 3x schedule, MS training."""

_base_ = "./mask_rcnn_hrnetv2p_w32_3x_ins_seg_bdd100k.py"
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w40",
    backbone=dict(
        type="HRNet",
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320)),
        ),
    ),
    neck=dict(type="HRFPN", in_channels=[40, 80, 160, 320], out_channels=256),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_hrnetv2p_w40_3x_ins_seg_bdd100k.pth"
