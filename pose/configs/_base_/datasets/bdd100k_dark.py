"""Dataset settings."""

_base_ = "./bdd100k.py"

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True, channel_order="bgr"),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(
        type="TopDownHalfBodyTransform",
        num_joints_half_body=8,
        prob_half_body=0.3,
    ),
    dict(
        type="TopDownGetRandomScaleRotation", rot_factor=40, scale_factor=0.5
    ),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(
        type="NormalizeTensor",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    dict(type="TopDownGenerateTarget", sigma=2, unbiased_encoding=True),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]

data = dict(train=dict(pipeline=train_pipeline))
