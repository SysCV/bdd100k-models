"""GCNet Mask RCNN with ResNet50-FPN, 1x schedule."""

_base_ = "./mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.py"
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(type="ContextBlock", ratio=1.0 / 4),
                stages=(False, True, True, True),
                position="after_conv3",
            )
        ]
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/gcnet_mask_rcnn_r50_fpn_1x_ins_seg_bdd100k.pth"
