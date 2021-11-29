# model settings
model = dict(
    type="TopDown",
    pretrained="https://download.openmmlab.com/mmpose/"
    "pretrain_models/hrnet_w48-8ef0771d.pth",
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
            ),
        ),
    ),
    keypoint_head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=48,
        out_channels=18,
        num_deconv_layers=0,
        extra=dict(
            final_conv_kernel=1,
        ),
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
