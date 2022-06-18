"""HRNet18, 3x schedule, MS training."""

_base_ = "./faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.py"
model = dict(
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144)),
        ),
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://msra/hrnetv2_w18"
        ),
    ),
    neck=dict(type="HRFPN", in_channels=[18, 36, 72, 144], out_channels=256),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w18_3x_det_bdd100k.pth"
