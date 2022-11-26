"""Dataset settings."""

dataset_type = "BDD100KDetDataset"  # pylint: disable=invalid-name
data_root = "../data/bdd100k/"  # pylint: disable=invalid-name # pylint: disable=invalid-name
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (720, 1280)
train_pipeline = [
    dict(type="Mosaic", img_scale=crop_size),
    dict(type="MixUp", img_scale=crop_size),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=(1280, 720), ratio_range=(0.5, 1.5)),
    dict(type="RandomCrop", crop_size=crop_size, allow_negative_crop=True),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/det_train_cocofmt.json",
        img_prefix=data_root + "images/100k/train",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=True,
    ),
    pipeline=train_pipeline,
)

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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/det_val_cocofmt.json",
        img_prefix=data_root + "images/100k/val",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "jsons/det_val_cocofmt.json",
        img_prefix=data_root + "images/100k/val",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
