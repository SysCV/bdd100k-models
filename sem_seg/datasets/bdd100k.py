"""Definition of the BDD100K dataset for semantic segmentation."""

import os.path as osp
from typing import List

import mmcv
import numpy as np
from mmseg.datasets import DATASETS, CustomDataset

from scalabel.label.io import save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Frame, Label


@DATASETS.register_module()
class BDD100KSemSegDataset(CustomDataset):  # type: ignore
    """BDD100K dataset for semantic segmentation."""

    CLASSES = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def format_results(  # pylint: disable=arguments-differ
        self, results: List[np.ndarray], out_dir: str  # type: ignore
    ) -> None:
        """Format the results into dir (standard format for BDD100K)."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: "
            f"{len(results)} != {len(self)}"
        )
        mmcv.mkdir_or_exist(out_dir)

        frames = []
        ann_id = 0
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]["filename"]
            frame = Frame(name=filename, labels=[])
            frames.append(frame)
            assert frame.labels is not None

            for pid in np.unique(result):
                ann_id += 1
                mask = (result == pid).astype(np.uint8)
                label = Label(
                    id=str(ann_id),
                    rle=mask_to_rle(mask),
                    category=self.CLASSES[pid],
                )
                frame.labels.append(label)

            prog_bar.update()

        save(osp.join(out_dir, "sem_seg.json"), frames)
