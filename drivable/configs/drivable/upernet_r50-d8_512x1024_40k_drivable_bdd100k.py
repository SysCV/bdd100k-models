"""UPerNet with ResNet-50-d8."""

_base_ = [
    "../_base_drive_/models/upernet_r50.py",
    "../_base_drive_/datasets/bdd100k_512x1024.py",
    "../_base_drive_/default_runtime.py",
    "../_base_drive_/schedules/schedule_40k.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/upernet_r50-d8_512x1024_40k_drivable_bdd100k.pth"
