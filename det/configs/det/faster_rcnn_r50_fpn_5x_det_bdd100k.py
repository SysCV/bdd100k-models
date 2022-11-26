"""Faster RCNN with ResNet50-FPN, 5x schedule, MS training."""

_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_5x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_5x_det_bdd100k.pth"
