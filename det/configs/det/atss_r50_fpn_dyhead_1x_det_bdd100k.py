"""ATSS with ResNet50-FPN and DyHead, 1x schedule."""

_base_ = [
    "../_base_/models/atss_r50_fpn_dyhead.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/atss_r50_fpn_dyhead_1x_det_bdd100k.pth"
