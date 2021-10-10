"""Definition of the BDD100K dataset for semantic segmentation."""

import os.path as osp
from typing import List

import mmcv
import numpy as np
from mmseg.datasets import DATASETS, CustomDataset
from PIL import Image


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
