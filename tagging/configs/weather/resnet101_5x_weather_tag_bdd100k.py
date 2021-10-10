"""ResNet101 Backbone.

- 640x640 crop
- 32 batch size (8 x 4)
- 5x schedule (60 epochs)
- 0.1 learning rate, step policy = (30, 45)
"""

_base_ = [
    "../_base_/models/resnet101.py",
    "../_base_/datasets/bdd100k_weather.py",
    "../_base_/schedules/schedule_5x.py",
    "../_base_/default_runtime.py",
]
data = dict(samples_per_gpu=8)
load_from = "https://dl.cv.ethz.ch/bdd100k/tagging/weather/models/resnet101_5x_weather_tag_bdd100k.pth"
