"""ResNet strikes back, RetinaNet with ResNet50-FPN, 1x schedule."""

_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
checkpoint = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb256-rsb-a1-600e_in1k_20211228-20e21305.pth"  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint=checkpoint,
        )
    )
)
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(norm_decay_mult=0.0, bypass_duplicate=True),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_rsb_1x_det_bdd100k.pth"
