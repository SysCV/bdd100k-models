"""HRNet18, 1x schedule."""

_base_ = "./mask_rcnn_hrnetv2p_w32_1x_ins_seg_bdd100k.py"
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w18",
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144)),
        ),
    ),
    neck=dict(type="HRFPN", in_channels=[18, 36, 72, 144], out_channels=256),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_hrnetv2p_w18_1x_ins_seg_bdd100k.pth"
