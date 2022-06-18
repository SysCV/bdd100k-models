"""Faster RCNN with Swin-S, 3x schedule, MS training."""

_base_ = "./faster_rcnn_swin-s_fpn_3x_det_bdd100k.py"

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_swin-s_fpn_fp16_3x_det_bdd100k.pth"
