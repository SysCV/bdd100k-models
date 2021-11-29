"""Definition of the BDD100K dataset for pose estimation."""

import os
from collections import defaultdict
from typing import Any, Dict, List

from mmpose.core.post_processing import oks_nms, soft_oks_nms
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.top_down.topdown_coco_dataset import (
    TopDownCocoDataset,
)
from scalabel.label.io import save
from scalabel.label.transforms import keypoints_to_nodes, nodes_to_edges
from scalabel.label.typing import Frame, Graph, Label

DictStrAny = Dict[str, Any]  # type: ignore


@DATASETS.register_module()
class TopDownBDD100KPoseDataset(TopDownCocoDataset):  # type: ignore
    """BDD100K dataset for top-down pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    BDD100K keypoint indexes::

        0: 'head',
        1: 'neck',
        2: 'right_shoulder',
        3: 'right_elbow',
        4: 'right_wrist',
        5: 'left_shoulder',
        6: 'left_elbow',
        7: 'left_wrist',
        8: 'right_hip',
        9: 'right_knee',
        10: 'right_ankle',
        11: 'left_hip',
        12: 'left_knee',
        13: 'left_ankle',
        14: 'right_hand',
        15: 'left_hand',
        16: 'right_foot',
        17: 'left_foot'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    EDGE_MAP = {
        0: ([1], "body"),
        1: ([2, 5, 8, 11], "body"),
        2: ([3], "right_side"),
        3: ([4], "right_side"),
        4: ([14], "right_side"),
        5: ([6], "left_side"),
        6: ([7], "left_side"),
        7: ([15], "left_side"),
        8: ([9], "right_side"),
        9: ([10], "right_side"),
        10: ([16], "right_side"),
        11: ([12], "left_side"),
        12: ([13], "left_side"),
        13: ([17], "left_side"),
    }
    KPT_NAMES = [
        "head",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hand",
        "left_hand",
        "right_foot",
        "left_foot",
    ]

    def convert_format(
        self, outputs: List[DictStrAny], res_folder: str
    ) -> None:
        """Format the results to the BDD100K prediction format."""
        res_file = os.path.join(res_folder, "result_keypoints.json")

        kpts = defaultdict(list)

        for output in outputs:
            preds = output["preds"]
            boxes = output["boxes"]
            image_paths = output["image_paths"]
            bbox_ids = output["bbox_ids"]

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix) :]]
                kpts[image_id].append(
                    {
                        "keypoints": preds[i],
                        "center": boxes[i][0:2],
                        "scale": boxes[i][2:4],
                        "area": boxes[i][4],
                        "score": boxes[i][5],
                        "image_id": image_id,
                        "bbox_id": bbox_ids[i],
                    }
                )
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        num_joints = self.ann_info["num_joints"]
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        valid_kpts = []
        for _, img_kpts in kpts.items():
            for n_p in img_kpts:
                box_score = n_p["score"]
                kpt_score = 0.0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p["keypoints"][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p["score"] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, oks_thr, sigmas=self.sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        self._write_bdd100k_keypoint_results(valid_kpts, res_file)

    def _write_bdd100k_keypoint_results(
        self, keypoints: List[List[DictStrAny]], res_file: str
    ) -> None:
        """Write results into BDD100K prediction format."""
        data_pack = [
            {
                "cat_id": self._class_to_coco_ind[cls],
                "cls_ind": cls_ind,
                "cls": cls,
                "ann_type": "keypoints",
                "keypoints": keypoints,
            }
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        pose_frames = [
            Frame(name=self.id2name[img_id], labels=[])
            for img_id in self.img_ids
        ]
        for i, res in enumerate(results):
            nodes = keypoints_to_nodes(res["keypoints"], self.KPT_NAMES)
            graph = Graph(
                nodes=nodes,
                edges=nodes_to_edges(nodes, self.EDGE_MAP),
                type="Pose2D-18Joints_Pred",
            )
            label = Label(
                id=i, score=res["score"], category="pedestrian", graph=graph
            )
            pose_frames[res["image_id"] - 1].labels.append(label)

        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        save(res_file, pose_frames)
