"""UPerNet with DeiT-S."""

_base_ = './upernet_vit-b_512x1024_80k_bdd100k.py'
model = dict(
    pretrained='pretrain/deit_small_patch16_224-cd65a155.pth',
    backbone=dict(num_heads=6, embed_dims=384, drop_path_rate=0.1),
    decode_head=dict(in_channels=[384, 384, 384, 384]),
    neck=dict(in_channels=[384, 384, 384, 384], out_channels=384),
    auxiliary_head=dict(in_channels=384))
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_deit-s_512x1024_80k_sem_seg_bdd100k.pth"
