# BDD100K Model Zoo

![teaser](./doc/images/teaser.png)

![visitors](https://visitor-badge.glitch.me/badge?page_id=SysCV.bdd100k-models&left_color=gray&right_color=blue)

In this repository, we provide popular models for each task in the [BDD100K dataset](https://www.bdd100k.com/). For each task in the dataset, we make publicly available the model weights, evaluation results, predictions, visualizations, as well as scripts to performance evaluation and visualization. The goal is to provide a set of competitive baselines to facilitate research and provide a common benchmark for comparison.

The number of pre-trained models in this zoo is :three::zero::zero:. **You can include your models in this repo as well!** See [contribution](./doc/CONTRIBUTING.md) instructions.

This repository currently supports the tasks listed below. For more information about each task, click on the task name. We plan to support all tasks in the BDD100K dataset eventually; see the [roadmap](#roadmap) for our plan and progress.

- [**Image Tagging**](./tagging)
- [**Object Detection**](./det)
- [**Instance Segmentation**](./ins_seg)
- [**Semantic Segmentation**](./sem_seg)
- [**Panoptic Segmentation**](./pan_seg)
- [**Drivable Area**](./drivable)
- [**Multiple Object Tracking (MOT)**](./mot)
- [**Multiple Object Tracking and Segmentation (MOTS)**](./mots)
- [**Pose Estimation**](./pose)

If you have any questions, please go to the BDD100K [discussions](https://github.com/bdd100k/bdd100k/discussions).

## Roadmap

- [x] Pose estimation
- [x] Panoptic segmentation
- [ ] Lane marking

## Dataset

Please refer to the [dataset preparation](./doc/PREPARE_DATASET.md) instructions for how to prepare and use the BDD100K dataset with the models.

## Maintainers

- [Thomas E. Huang](https://thomasehuang.com/) [@thomasehuang](https://github.com/thomasehuang)

## Citation

To cite the BDD100K dataset in your paper,

```latex
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```
