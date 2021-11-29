"""HRNet32, 1x schedule."""

_base_ = "./mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.py"
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
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_hrnetv2p_w32_1x_ins_seg_bdd100k.pth"
