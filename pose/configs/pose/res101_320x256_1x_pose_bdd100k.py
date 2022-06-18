"""ResNet101, 1x schedule."""

_base_ = "./res50_320x256_1x_pose_bdd100k.py"
model = dict(pretrained="torchvision://resnet101", backbone=dict(depth=101))
load_from = "https://dl.cv.ethz.ch/bdd100k/pose/models/res101_320x256_1x_pose_bdd100k.pth"
