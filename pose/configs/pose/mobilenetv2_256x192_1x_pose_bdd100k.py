"""Mobilenetv2, 1x schedule."""

_base_ = [
    "../_base_/models/mobilenetv2.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/pose/models/mobilenetv2_256x192_1x_pose_bdd100k.pth"
