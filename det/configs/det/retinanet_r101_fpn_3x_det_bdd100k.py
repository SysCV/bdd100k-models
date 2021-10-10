"""RetinaNet with ResNet101-FPN, 3x schedule, MS training."""

_base_ = "./retinanet_r50_fpn_3x_det_bdd100k.py"
model = dict(pretrained="torchvision://resnet101", backbone=dict(depth=101))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_r101_fpn_3x_det_bdd100k.pth"
