"""HRNet48 with a simple convolution head."""

_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
model = dict(
    pretrained="open-mmlab://msra/hrnetv2_w48",
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)),
        )
    ),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])
    ),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/fcn_hr48_512x1024_80k_drivable_bdd100k.pth"
