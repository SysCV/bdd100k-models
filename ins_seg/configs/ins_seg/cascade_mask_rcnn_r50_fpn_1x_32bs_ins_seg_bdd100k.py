"""Cascade Mask RCNN with ResNet50-FPN, 1x schedule, 32 batch size."""

_base_ = [
    "../_base_/models/cascade_mask_rcnn_r50_fpn.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
data = dict(samples_per_gpu=4, workers_per_gpu=4)
optimizer = dict(type="SGD", lr=0.04, momentum=0.9, weight_decay=0.0001)
load_from = "https://dl.cv.ethz.ch/bdd100k/ins_seg/models/cascade_mask_rcnn_r50_fpn_1x_32bs_ins_seg_bdd100k.pth"
