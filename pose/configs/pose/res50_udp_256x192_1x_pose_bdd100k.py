"""ResNet50, 1x schedule."""

_base_ = [
    "../_base_/models/res50.py",
    "../_base_/datasets/bdd100k_udp.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
target_type = "GaussianHeatmap"
model = dict(
    test_cfg=dict(shift_heatmap=False, target_type=target_type, use_udp=True)
)
load_from = "https://dl.cv.ethz.ch/bdd100k/pose/models/res50_udp_256x192_1x_pose_bdd100k.pth"
