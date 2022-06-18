"""RetinaNet with PVTv2-B1, 3x schedule, MS training."""

_base_ = './retinanet_pvtv2-b0_fpn_3x_bdd100k.py'
model = dict(
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_pvtv2-b2_fpn_3x_det_bdd100k.pth"
