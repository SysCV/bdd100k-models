"""Definition of the BDD100K dataset for drivable area."""

import os.path as osp
from typing import List

import mmcv
import numpy as np
from mmseg.datasets import DATASETS, CustomDataset

from scalabel.label.io import save
from scalabel.label.transforms import mask_to_rle
from scalabel.label.typing import Frame, Label


@DATASETS.register_module()
class BDD100KDrivableDataset(CustomDataset):  # type: ignore
    """BDD100K dataset for drivable area."""

    CLASSES = ("direct", "alternative", "background")
    PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]

    def format_results(  # pylint: disable=arguments-differ
        self, results: List[np.ndarray], out_dir: str  # type: ignore
    ) -> Nonegit :
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

        save(osp.join(out_dir, "drivable.json"), frames)
