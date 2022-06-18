"""RetinaNet with PVT-S, 3x schedule, MS training."""

_base_ = './retinanet_pvt-t_fpn_3x_bdd100k.py'
model = dict(
    backbone=dict(
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_small.pth')))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_pvt-s_fpn_3x_det_bdd100k.pth"
