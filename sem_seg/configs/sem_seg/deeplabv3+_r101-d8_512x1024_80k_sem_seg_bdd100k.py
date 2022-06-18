"""DeeplabV3+ with ResNet-101-d8."""

_base_ = "./deeplabv3+_r50-d8_512x1024_80k_sem_seg_bdd100k.py"
model = dict(pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101))
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/deeplabv3+_r101-d8_512x1024_80k_sem_seg_bdd100k.pth"
