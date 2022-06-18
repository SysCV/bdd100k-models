"""Sparse RCNN with ResNet50-FPN, 100 proposals, 3x schedule, MS training."""

_base_ = "./sparse_rcnn_r50_fpn_1x_det_bdd100k.py"
data_root = "../data/bdd100k/"  # pylint: disable=invalid-name
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
min_values = (600, 624, 648, 672, 696, 720)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[(1280, value) for value in min_values],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

data = dict(train=dict(pipeline=train_pipeline))
lr_config = dict(policy="step", step=[27, 33])
runner = dict(type="EpochBasedRunner", max_epochs=36)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/sparse_rcnn_r50_fpn_3x_det_bdd100k.pth"
