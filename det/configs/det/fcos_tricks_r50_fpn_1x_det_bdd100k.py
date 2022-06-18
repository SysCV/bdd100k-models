"""FCOS with ResNet50-FPN, 1x schedule, tricks."""

_base_ = "./fcos_r50_fpn_1x_det_bdd100k.py"

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="open-mmlab://detectron2/resnet50_caffe",
        )
    ),
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type="GIoULoss", loss_weight=1.0),
    ),
    # training and testing settings
    test_cfg=dict(nms=dict(type="nms", iou_threshold=0.6)),
)

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(warmup="linear")
load_from = "https://dl.cv.ethz.ch/bdd100k/det/models/fcos_tricks_r50_fpn_1x_det_bdd100k.pth"
