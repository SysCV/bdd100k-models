"""Definition of the BDD100K dataset."""

import os
import os.path as osp
from typing import List

import numpy as np
from scalabel.label.io import save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Frame, Label

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets import DATASETS, CocoPanopticDataset

SHAPE = [720, 1280]


@DATASETS.register_module()
class BDD100KPanSegDataset(CocoPanopticDataset):  # type: ignore
    """BDD100K Panoptic Dataset."""

    CLASSES = [
        "person", "rider", "bicycle", "bus", "car", "caravan", "motorcycle", "trailer", "train", "truck", "dynamic", "ego vehicle", "ground", "static", "parking", "rail track", "road", "sidewalk", "bridge", "building", "fence", "garage", "guard rail", "tunnel", "wall", "banner", "billboard", "lane divider", "parking sign", "pole", "polegroup", "street light", "traffic cone", "traffic device", "traffic light", "traffic sign", "traffic sign frame", "terrain", "vegetation", "sky",
    ]
    STUFF_CLASSES = ["dynamic", "ego vehicle", "ground", "static", "parking", "rail track", "road", "sidewalk", "bridge", "building", "fence", "garage", "guard rail", "tunnel", "wall", "banner", "billboard", "lane divider", "parking sign", "pole", "polegroup", "street light", "traffic cone", "traffic device", "traffic light", "traffic sign", "traffic sign frame", "terrain", "vegetation", "sky"]
    THING_CLASSES = ["person", "rider", "bicycle", "bus", "car", "caravan", "motorcycle", "trailer", "train", "truck"]

    def convert_format(  # pylint: disable=arguments-differ
        self, results: List[np.ndarray], out_dir: str  # type: ignore
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"Length of res and dset not equal: {len(results)} != {len(self)}"
        os.makedirs(out_dir, exist_ok=True)

        frames = []
        label2cat = dict((v, k) for (k, v) in self.cat2label.items())
        for img_idx in range(len(self)):
            img_name = self.data_infos[img_idx]["file_name"]
            frame = Frame(name=img_name, labels=[])
            frames.append(frame)
            assert frame.labels is not None

            pan_results = results[img_idx]['pan_results']
            pan_labels = np.unique(pan_results)
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                # We reserve the length of self.CLASSES for VOID label
                if sem_label == len(self.CLASSES):
                    continue
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                mask = pan_results == pan_label
                label = Label(
                    id=str(pan_label),
                    rle=mask_to_rle(mask),
                    category=self.categories[cat_id]["name"],
                )
                frame.labels.append(label)

        save(osp.join(out_dir, "pan_seg.json"), frames)
