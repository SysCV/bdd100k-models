"""UPerNet with ConvNeXt-S."""

_base_ = "./upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py"

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(in_channels=[96, 192, 384, 768]),
    auxiliary_head=dict(in_channels=384)
)
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_convnext-s_fp16_512x1024_80k_sem_seg_bdd100k.pth"
