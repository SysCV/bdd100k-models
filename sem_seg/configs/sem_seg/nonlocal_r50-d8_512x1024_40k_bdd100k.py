"""NonLocal with ResNet-50-d8."""

_base_ = [
    "../_base_/models/nonlocal_r50-d8.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
