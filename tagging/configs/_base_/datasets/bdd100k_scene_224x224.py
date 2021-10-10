# BDD100K Scene Tagging dataset with size 224x224
_base_ = "./bdd100k_weather_224x224.py"  # pylint: disable=invalid-name
dataset_type = "BDD100KSceneTaggingDataset"  # pylint: disable=invalid-name
data = dict(
    train=dict(type=dataset_type),
    val=dict(type=dataset_type),
    test=dict(type=dataset_type),
)
