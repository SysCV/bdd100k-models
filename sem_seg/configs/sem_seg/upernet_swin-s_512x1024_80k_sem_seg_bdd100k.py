"""UPerNet with Swin-S."""

_base_ = "./upernet_swin-t_512x1024_80k_sem_seg_bdd100k.py"

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=19),
    auxiliary_head=dict(in_channels=384, num_classes=19))
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_swin-s_512x1024_80k_sem_seg_bdd100k.pth"
