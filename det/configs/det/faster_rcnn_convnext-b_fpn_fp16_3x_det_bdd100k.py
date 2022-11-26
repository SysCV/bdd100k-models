"""Faster RCNN with ConvNeXt-B, 3x schedule, MS training, FP16."""

_base_ = "./faster_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.py"

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
        neck=dict(in_channels=[128, 256, 512, 1024]))
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_convnext-b_fpn_fp16_3x_det_bdd100k.pth"
