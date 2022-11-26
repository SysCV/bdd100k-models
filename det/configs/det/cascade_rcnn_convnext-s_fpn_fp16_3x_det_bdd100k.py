"""Cascade RCNN with ConvNeXt-S, 3x schedule, MS training, FP16."""

_base_ = [
    "../_base_/models/cascade_rcnn_r50_fpn.py",
    "../_base_/datasets/bdd100k_mstrain.py",
    "../_base_/schedules/schedule_3x.py",
    "../_base_/default_runtime.py",
]

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 12
    })
lr_config = dict(warmup_iters=1000, step=[27, 33])

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/cascade_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.pth"
