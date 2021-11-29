# model settings
model = dict(
    type="TopDown",
    pretrained="torchvision://resnet50",
    backbone=dict(type="ResNet", depth=50),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=2048,
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
