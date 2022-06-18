"""Dataset settings."""

dataset_type = "BDD100KPanSegDataset"  # pylint: disable=invalid-name
data_root = "../data/bdd100k/"  # pylint: disable=invalid-name
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(
        type="LoadPanopticAnnotations",
        with_bbox=True,
        with_mask=True,
        with_seg=True,
    ),
    dict(type="Resize", img_scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="SegRescale", scale_factor=1 / 4),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_masks", "gt_semantic_seg"],
    ),
]
test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
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
        ann_file=data_root + "jsons/pan_seg_train_cocofmt.json",
        seg_prefix=data_root + "jsons/pan_seg/masks/train/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/pan_seg_val_cocofmt.json",
        seg_prefix=data_root + "jsons/pan_seg/masks/val/",
        ins_ann_file=data_root + "jsons/ins_seg_val_cocofmt.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/pan_seg_val_cocofmt.json",
        seg_prefix=data_root + "jsons/pan_seg/masks/val/",
        ins_ann_file=data_root + "jsons/ins_seg_val_cocofmt.json",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric=["PQ"])
