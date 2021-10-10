"""ResNet50 Backbone.

- 640x640 crop
- 48 batch size (12 x 4)
- 4x schedule (48 epochs)
- 0.1 learning rate, step policy = (24, 36)
"""

_base_ = [
    "../_base_/models/resnet50.py",
    "../_base_/datasets/bdd100k_scene.py",
    "../_base_/schedules/schedule_4x.py",
    "../_base_/default_runtime.py",
]
load_from = "https://dl.cv.ethz.ch/bdd100k/tagging/scene/models/resnet50_4x_scene_tag_bdd100k.pth"
