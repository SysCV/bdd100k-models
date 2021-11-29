# model settings
model = dict(
    type="TopDown",
    pretrained="mmcls://mobilenet_v2",
    backbone=dict(type="MobileNetV2", widen_factor=1.0, out_indices=(7,)),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1280,
        out_channels=18,
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process="default",
        shift_heatmap=True,
        modulate_kernel=11,
    ),
)
