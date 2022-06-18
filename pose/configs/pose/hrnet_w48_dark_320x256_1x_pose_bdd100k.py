"""HRNet-w48, 1x schedule."""

_base_ = [
    "../_base_/models/hrnet_w48.py",
    "../_base_/datasets/bdd100k_dark.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
data_cfg = dict(image_size=[256, 320], heatmap_size=[64, 80])
data = dict(
    train=dict(data_cfg=data_cfg),
    val=dict(data_cfg=data_cfg),
    test=dict(data_cfg=data_cfg),
)
model = dict(test_cfg=dict(post_process="unbiased"))
load_from = "https://dl.cv.ethz.ch/bdd100k/pose/models/hrnet_w48_dark_320x256_1x_pose_bdd100k.pth"
