# model settings
_base_ = "./mask_rcnn_r50_fpn.py"
model = dict(
    type="PanopticFPN",
    roi_head=dict(
        bbox_head=dict(num_classes=10), mask_head=dict(num_classes=10)
    ),
    semantic_head=dict(
        type="PanopticFPNHead",
        num_things_classes=10,
        num_stuff_classes=30,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(
            type="CrossEntropyLoss", ignore_index=255, loss_weight=0.5
        ),
    ),
    panoptic_fusion_head=dict(
        type="HeuristicFusionHead", num_things_classes=10, num_stuff_classes=30
    ),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type="nms", iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096,
        )
    ),
)
