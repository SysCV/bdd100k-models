"""Definition of the BDD100K dataset."""

import json
import os
import os.path as osp
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import pycocotools.mask as mask_utils
from mmdet.datasets import DATASETS, CocoDataset
from PIL import Image
from scalabel.label.coco_typing import RLEType
from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Frame, Label
from tqdm import tqdm

SHAPE = [720, 1280]


def mask_merge(
    img_name: str,
    scores: List[float],
    segms: List[np.ndarray],  # type: ignore
    colors: List[List[int]],
    bitmask_base: str,
) -> None:
    """Merge masks into a bitmask png file."""
    bitmask = np.zeros((*SHAPE, 4), dtype=np.uint8)
    sorted_idxs = np.argsort(scores)
    for idx in sorted_idxs:
        mask = mask_utils.decode(segms[idx])
        for i in range(4):
            bitmask[..., i] = (
                bitmask[..., i] * (1 - mask) + mask * colors[idx][i]
            )
    bitmask_path = osp.join(bitmask_base, img_name.replace(".jpg", ".png"))
    bitmask_dir = osp.split(bitmask_path)[0]
    if not osp.exists(bitmask_dir):
        os.makedirs(bitmask_dir)
    bitmask_pil = Image.fromarray(bitmask)
    bitmask_pil.save(bitmask_path)


def mask_merge_parallel(
    bitmask_base: str,
    img_names: List[str],
    scores_list: List[List[float]],
    segms_list: List[List[RLEType]],
    colors_list: List[List[List[int]]],
    nproc: int = 4,
) -> None:
    """Merge masks into a bitmask png file. Run parallely."""
    with Pool(nproc) as pool:
        print("\nMerging overlapped masks.")
        pool.starmap(
            partial(mask_merge, bitmask_base=bitmask_base),
            tqdm(
                zip(img_names, scores_list, segms_list, colors_list),
                total=len(img_names),
            ),
        )


@DATASETS.register_module()
class BDD100KInsSegDataset(CocoDataset):  # type: ignore
    """BDD100K Dataset."""

    CLASSES = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    def convert_format(  # pylint: disable=arguments-differ
        self, results: List[np.ndarray], out_dir: str  # type: ignore
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"Length of res and dset not equal: {len(results)} != {len(self)}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        scores_list, segms_list, colors_list = [], [], []
        det_frames, seg_frames = [], []
        img_names = []
        ann_id = 0

        for img_idx in range(len(self)):
            index = 0
            img_name = self.data_infos[img_idx]["file_name"]
            img_names.append(img_name)
            det_frame = Frame(name=img_name, labels=[])
            det_frames.append(det_frame)
            seg_frame = Frame(name=img_name, labels=[])
            seg_frames.append(seg_frame)
            scores, segms, colors = [], [], []

            det_results, seg_results = results[img_idx]
            for cat_idx, [cur_det, cur_seg] in enumerate(
                zip(det_results, seg_results)
            ):
                for bbox, segm in zip(cur_det, cur_seg):
                    ann_id += 1
                    index += 1
                    score = bbox[-1]

                    det_label = Label(
                        id=str(ann_id),
                        score=score,
                        box2d=bbox_to_box2d(self.xyxy2xywh(bbox)),
                        category=self.CLASSES[cat_idx],
                    )
                    det_frame.labels.append(det_label)  # type: ignore

                    seg_label = Label(id=str(ann_id), index=index, score=score)
                    seg_frame.labels.append(seg_label)  # type: ignore

                    scores.append(score)
                    segms.append(segm)
                    colors.append([cat_idx + 1, 0, index >> 8, index & 255])

            scores_list.append(scores)
            segms_list.append(segms)
            colors_list.append(colors)

        det_out_path = osp.join(out_dir, "det.json")
        save(det_out_path, det_frames)

        seg_out_path = osp.join(out_dir, "score.json")
        seg_frame_dicts = [seg_frame.dict() for seg_frame in seg_frames]
        with open(seg_out_path, "w", encoding="utf-8") as fp:
            json.dump(seg_frame_dicts, fp, indent=2)

        bitmask_dir = osp.join(out_dir, "bitmasks")
        mask_merge_parallel(
            bitmask_dir,
            img_names,
            scores_list,
            segms_list,
            colors_list,
            nproc=4,
        )
