"""Mask RCNN with ResNet50-FPN, 3x schedule, MS training."""

_base_ = [
    "../_base_/models/mask_rcnn_r50_fpn.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.pth"
