## Prepare Datasets

### Download BDD100K
Please first download the images and annotations from the [official website](https://bdd-data.berkeley.edu/).
For more details about the dataset, please refer to the [official documentation](https://doc.bdd100k.com/download.html).

On the official download page, the required data and annotations for each task are:

- `image tagging` set:
  - images: `100K Images`
  - annotations: `Detection 2020 Labels`
- `object detection` set:
  - images: `100K Images`
  - annotations: `Detection 2020 Labels`
- `pose estimation` set:
  - images: `100K Images`
  - annotations: `Pose Estimation Labels`
- `instance segmentation` set:
  - images: `10K Images`
  - annotations: `Instance Segmentation`
- `semantic segmentation` set:
  - images: `10K Images`
  - annotations: `Semantic Segmentation`
- `drivable area` set:
  - images: `100K Images`
  - annotations: `Drivable Area`
- `box tracking (MOT)` set:
  - images: `MOT 2020 Images`
  - annotations: `MOT 2020 Labels`
- `segmentation tracking (MOTS)` set:
  - images: `MOTS 2020 Images`
  - annotations: `MOTS 2020 Labels`

We list all the tasks here for completeness, but you only need to download the images and labels for the task you are interested in.

### Convert Annotations

For object detection and instance segmentation, please transform the official annotation files to COCO style with the provided [scripts](https://doc.bdd100k.com/format.html#to-coco) by BDD100K.

First, uncompress the downloaded annotation file and you will obtain a folder named `bdd100k`.

To convert the detection set, you can run:
```bash
mkdir bdd100k/jsons
python -m bdd100k.label.to_coco -m det \
    -i bdd100k/labels/det_20/det_${SET_NAME}.json \
    -o bdd100k/jsons/det_${SET_NAME}_cocofmt.json
```

To convert the pose estimation set, you can run:
```bash
mkdir bdd100k/jsons
python -m bdd100k.label.to_coco -m pose \
    -i bdd100k/labels/pose_21/pose_${SET_NAME}.json \
    -o bdd100k/jsons/pose_${SET_NAME}_cocofmt.json
```

To convert the instance segmentation set, you can run:
```bash
mkdir bdd100k/jsons
python -m bdd100k.label.to_coco -m ins_seg --only-mask \
    -i bdd100k/labels/ins_seg/bitmasks/${SET_NAME} \
    -o bdd100k/jsons/ins_seg_${SET_NAME}_cocofmt.json \
    [--nproc ${NUM_PROCESS}]
```

For box and segmentation tracking, you can also convert the annotations for each video to one COCO style JSON annotation file by running:

To convert the box tracking set, you can run:
```bash
mkdir bdd100k/jsons
python -m bdd100k.label.to_coco -m box_track \
    -i bdd100k/labels/box_track_20/${SET_NAME} \
    -o bdd100k/jsons/box_track_${SET_NAME}_cocofmt.json
```

To convert the segmentation tracking set, you can run:
```bash
mkdir bdd100k/jsons
python -m bdd100k.label.to_coco -m seg_track \
    -i bdd100k/labels/seg_track_20/bitmasks/${SET_NAME} \
    -o bdd100k/jsons/seg_track_${SET_NAME}_cocofmt.json \
    [--nproc ${NUM_PROCESS}]
```

The `${SET_NAME}` here can be one of `['train', 'val']`.

### Symlink the Data

It is recommended to symlink the dataset root to `$bdd100-models/data`.
If your folder structure is different, you may need to change the corresponding paths in each config file, which is not recommended.
Our full folder structure is as follows:

```
bdd100k-models
└── data
    └── bdd100k
        ├── images
        │   ├── 100k
        |   |   ├── train
        |   |   └── val
        │   ├── 10k
        |   |   ├── train
        |   |   └── val
        |   ├── track
        |   |   ├── train
        |   |   ├── val
        |   |   └── test
        |   └── seg_track_20
        |       ├── train
        |       ├── val
        |       └── test
        ├── labels
        │   ├── det_20
        |   |   ├── det_train.json
        |   |   └── det_val.json
        │   ├── pose_21
        |   |   ├── pose_train.json
        |   |   └── pose_val.json
        │   ├── ins_seg
        |   |   ├── bitmasks
        |   |   |  ├── train
        |   |   |  └── val
        |   |   ├── colormaps
        |   |   └── polygons
        │   ├── sem_seg
        |   |   ├── masks
        |   |   |  ├── train
        |   |   |  └── val
        |   |   ├── colormaps
        |   |   └── polygons
        │   ├── drivable
        |   |   ├── masks
        |   |   |  ├── train
        |   |   |  └── val
        |   |   ├── colormaps
        |   |   └── polygons
        |   ├── box_track_20
        |   |   ├── train
        |   |   └── val
        |   └── seg_track_20
        |       ├── bitmasks
        |       |   ├── train
        |       |   └── val
        |       ├── colormaps
        |       |   ├── train
        |       |   └── val
        |       └── polygons
        |           ├── train
        |           └── val
        └── jsons
            ├── det_train_cocofmt.json
            ├── det_val_cocofmt.json
            ├── pose_train_cocofmt.json
            ├── pose_val_cocofmt.json
            ├── ins_seg_train_cocofmt.json
            ├── ins_seg_val_cocofmt.json
            ├── box_track_train_cocofmt.json
            ├── box_track_val_cocofmt.json
            ├── seg_track_train_cocofmt.json
            └── seg_track_val_cocofmt.json
```
