"""Definition of the BDD100K dataset."""

import json
from typing import Collection, Dict, List

import numpy as np
from mmcls.datasets import DATASETS, BaseDataset


def load_annotations(
    ann_file: str, data_prefix: str, classes: List[str], attr: str
) -> List[Dict[str, Collection[str]]]:
    """Load annotations from file."""
    assert isinstance(ann_file, str)

    data_infos = []
    with open(ann_file, encoding="utf-8") as f:
        labels = json.load(f)
        for label in labels:
            info = {
                "img_prefix": data_prefix,
                "img_info": {"filename": label["name"]},
                "gt_label": np.array(
                    classes.index(label["attributes"][attr]),
                    dtype=np.int64,
                ),
            }
            data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class BDD100KWeatherTaggingDataset(BaseDataset):  # type: ignore
    """BDD100K Dataset for image tagging."""

    CLASSES = [
        "rainy",
        "snowy",
        "clear",
        "overcast",
        "undefined",
        "partly cloudy",
        "foggy",
    ]

    def load_annotations(self) -> List[Dict[str, Collection[str]]]:
        """Load annotations from file."""
        return load_annotations(
            self.ann_file, self.data_prefix, self.CLASSES, "weather"
        )


@DATASETS.register_module()
class BDD100KSceneTaggingDataset(BaseDataset):  # type: ignore
    """BDD100K Dataset for image tagging."""

    CLASSES = [
        "tunnel",
        "residential",
        "parking lot",
        "undefined",
        "city street",
        "gas stations",
        "highway",
    ]

    def load_annotations(self) -> List[Dict[str, Collection[str]]]:
        """Load annotations from file."""
        return load_annotations(
            self.ann_file, self.data_prefix, self.CLASSES, "scene"
        )
