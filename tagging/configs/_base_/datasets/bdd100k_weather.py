# BDD100K Weather Tagging dataset with size 640x640
dataset_type = "BDD100KWeatherTaggingDataset"  # pylint: disable=invalid-name
data_root = "../data/bdd100k/"  # pylint: disable=invalid-name
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="RandomCrop", size=640),
    dict(type="RandomFlip", flip_prob=0.5, direction="horizontal"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="CenterCrop", crop_size=640),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "labels/bdd100k_labels_images_train.json",
        data_prefix=data_root + "images/100k/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "labels/bdd100k_labels_images_val.json",
        data_prefix=data_root + "images/100k/val",
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "labels/bdd100k_labels_images_val.json",
        data_prefix=data_root + "images/100k/val",
        pipeline=test_pipeline,
        test_mode=True,
    ),
)
