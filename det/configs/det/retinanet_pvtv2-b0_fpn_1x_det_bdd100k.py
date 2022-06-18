"""RetinaNet with PVTv2-B0, 1x schedule."""

_base_ = [
    "../_base_/models/retinanet_r50_fpn.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b0.pth')),
    neck=dict(in_channels=[32, 64, 160, 256]))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/retinanet_pvtv2-b0_fpn_1x_det_bdd100k.pth"
