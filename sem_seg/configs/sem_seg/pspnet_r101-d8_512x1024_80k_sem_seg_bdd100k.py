"""PSPNet with ResNet-101-d8."""

_base_ = "./pspnet_r50-d8_512x1024_80k_sem_seg_bdd100k.py"
model = dict(pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101))
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r101-d8_512x1024_80k_sem_seg_bdd100k.pth"
