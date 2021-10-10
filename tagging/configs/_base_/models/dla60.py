# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="DLA",
        levels=[1, 1, 1, 2, 3, 1],
        channels=[16, 32, 128, 256, 512, 1024],
        num_classes=7,
        block="Bottleneck",
    ),
    neck=None,
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
