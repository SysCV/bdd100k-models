"""PointRend with ResNet-50."""

_base_ = [
    "../_base_/models/pointrend_r50.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
lr_config = dict(warmup="linear", warmup_iters=200)
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/pointrend_r50_512x1024_40k_drivable_bdd100k.pth"
