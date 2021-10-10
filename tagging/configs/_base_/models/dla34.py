# model settings
model = dict(
    type="ImageClassifier",
    backbone=dict(
        type="DLA",
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        num_classes=7,
        block="BasicBlock",
    ),
    neck=None,
    head=dict(
        type="ClsHead",
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    ),
)
