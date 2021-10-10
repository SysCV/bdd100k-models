"""Cascade RCNN with ResNet101-FPN, 3x schedule, MS training."""

_base_ = "./cascade_rcnn_r50_fpn_3x_det_bdd100k.py"
model = dict(pretrained="torchvision://resnet101", backbone=dict(depth=101))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_r101_fpn_3x_det_bdd100k.pth"
