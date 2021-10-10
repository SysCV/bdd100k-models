"""Definition of the BDD100K dataset for drivable area."""

import os.path as osp
from typing import List

import mmcv
import numpy as np
from mmseg.datasets import DATASETS, CustomDataset
from PIL import Image


@DATASETS.register_module()
class BDD100KDrivableDataset(CustomDataset):  # type: ignore
    """BDD100K dataset for drivable area."""

    CLASSES = (
        "direct",
        "alternative",
        "background",
    )

    PALETTE = [
        [219, 94, 86],
        [86, 211, 219],
        [0, 0, 0],
    ]

    def results2img(
        self, results: List[np.ndarray], imgfile_prefix: str  # type: ignore
    ) -> List[str]:
        """Write the segmentation results to images."""
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]["filename"]
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f"{basename}.png")

            output = Image.fromarray(result.astype(np.uint8)).convert("P")
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(  # pylint: disable=arguments-differ
        self, results: List[np.ndarray], imgfile_prefix: str  # type: ignore
    ) -> List[str]:
        """Format the results into dir (standard format for BDD100K)."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: "
            f"{len(results)} != {len(self)}"
        )
        result_files = self.results2img(results, imgfile_prefix)

        return result_files
