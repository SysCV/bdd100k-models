"""GCNet with ResNet-50-d8."""

_base_ = [
    ".../_base_drive_/models/gcnet_r50-d8.py",
    ".../_base_drive_/datasets/bdd100k.py",
    ".../_base_drive_/default_runtime.py",
    ".../_base_drive_/schedules/schedule_40k.py",
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode="slide", crop_size=(769, 769), stride=(513, 513)),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/gcnet_r50-d8_769x769_40k_drivable_bdd100k.pth"
