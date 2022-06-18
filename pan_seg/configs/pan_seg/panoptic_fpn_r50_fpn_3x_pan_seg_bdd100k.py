"""Panoptic FPN with ResNet50-FPN, 3x schedule, MS training."""

_base_ = [
    "../_base_/models/panoptic_fpn_r50_fpn.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]

custom_hooks = []
load_from = "https://dl.cv.ethz.ch/bdd100k/pan_seg/models/panoptic_fpn_r50_fpn_3x_pan_seg_bdd100k.pth"
