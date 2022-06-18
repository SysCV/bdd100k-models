"""UPerNet with ConvNeXt-B."""

_base_ = "./upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py"

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/upernet_convnext-b_fp16_512x1024_80k_sem_seg_bdd100k.pth"
