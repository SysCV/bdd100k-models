"""FPN with ResNet-50."""

_base_ = [
    "../_base_/models/fpn_r50.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
model = dict(
    decode_head=dict(
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True)
    )
)
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/fpn_r50_gn_512x1024_80k_sem_seg_bdd100k.pth"
