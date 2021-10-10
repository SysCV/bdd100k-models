"""EMANet with ResNet-50-d8."""

_base_ = [
    "../_base_/models/emanet_r50-d8.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode="slide", crop_size=(769, 769), stride=(513, 513)),
)
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/emanet_r50-d8_769x769_80k_sem_seg_bdd100k.pth"
