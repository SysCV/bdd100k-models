"""VGG16 Backbone.

- 224x224 crop
- 128 batch size (32 x 4)
- 5x schedule (60 epochs)
- 0.1 learning rate, step policy = (30, 45)
"""

_base_ = [
    "../_base_/models/vgg16.py",
    "../_base_/datasets/bdd100k_scene_224x224.py",
    "../_base_/schedules/schedule_5x.py",
    "../_base_/default_runtime.py",
]
optimizer = dict(lr=0.01)
load_from = "https://dl.cv.ethz.ch/bdd100k/tagging/scene/models/vgg16_5x_224x224_scene_tag_bdd100k.pth"
