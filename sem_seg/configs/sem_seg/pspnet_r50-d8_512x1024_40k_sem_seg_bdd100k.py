"""PSPNet with ResNet-50-d8."""

_base_ = [
    "../_base_/models/pspnet_r50-d8.py",
    "../_base_/datasets/bdd100k_512x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/sem_seg/models/pspnet_r50-d8_512x1024_40k_sem_seg_bdd100k.pth"
