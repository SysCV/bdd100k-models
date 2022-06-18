"""UPerNet with Swin-B, FP16."""

_base_ = "./upernet_swin-s_512x1024_80k_sem_seg_bdd100k.py"

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19))

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_swin-b_fp16_512x1024_80k_sem_seg_bdd100k.pth"
