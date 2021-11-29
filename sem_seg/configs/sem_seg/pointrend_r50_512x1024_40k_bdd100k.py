"""PointRend with ResNet-50-FPN."""

_base_ = [
    "../_base_/models/pointrend_r50.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
lr_config = dict(warmup="linear", warmup_iters=200)
