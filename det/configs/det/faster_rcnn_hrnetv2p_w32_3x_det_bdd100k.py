"""HRNet32, 3x schedule, MS training."""

_base_ = "./faster_rcnn_r50_fpn_3x_det_bdd100k.py"
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w32",
    backbone=dict(
        _delete_=True,
        type="HRNet",
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(32, 64),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
            ),
        ),
    ),
    neck=dict(
        _delete_=True,
        type="HRFPN",
        in_channels=[32, 64, 128, 256],
        out_channels=256,
    ),
)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_hrnetv2p_w32_3x_det_bdd100k.pth"
