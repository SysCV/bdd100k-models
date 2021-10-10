"""Dataset settings."""

dataset_type = "BDD100KInsSegDataset"  # pylint: disable=invalid-name
data_root = "../data/bdd100k/"  # pylint: disable=invalid-name
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        img_scale=[
            (1280, 600),
            (1280, 624),
            (1280, 648),
            (1280, 672),
            (1280, 696),
            (1280, 720),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/ins_seg_train_cocofmt.json",
        img_prefix=data_root + "images/10k/val",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/ins_seg_val_cocofmt.json",
        img_prefix=data_root + "images/10k/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/ins_seg_val_cocofmt.json",
        img_prefix=data_root + "images/10k/val",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric=["bbox", "segm"])
