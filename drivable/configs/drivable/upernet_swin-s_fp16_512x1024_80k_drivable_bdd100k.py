"""UPerNet with Swin-S, FP16."""

_base_ = "./upernet_swin-s_512x1024_80k_drivable_bdd100k.py"

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
load_from = "https://dl.cv.ethz.ch/bdd100k/drivable/models/upernet_swin-s_fp16_512x1024_80k_drivable_bdd100k.pth"
