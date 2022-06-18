"""ATSS with ResNet50-FPN and DyHead, 3x schedule, MS training."""

_base_ = [
    "../_base_/models/atss_r50_fpn_dyhead.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/atss_r50_fpn_dyhead_3x_det_bdd100k.pth"
