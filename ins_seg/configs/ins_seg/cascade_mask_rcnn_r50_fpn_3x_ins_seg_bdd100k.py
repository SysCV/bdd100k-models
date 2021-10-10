"""Cascade Mask RCNN with ResNet50-FPN, 3x schedule."""

_base_ = [
    "../_base_/models/cascade_mask_rcnn_r50_fpn.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/cascade_mask_rcnn_r50_fpn_3x_ins_seg_bdd100k.pth"
