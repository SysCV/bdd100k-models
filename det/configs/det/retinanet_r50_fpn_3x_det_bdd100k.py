"""RetinaNet with ResNet50-FPN, 3x schedule, MS training."""

_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r50_fpn_3x_det_bdd100k.pth"
