"""Inference a pretrained model."""

import argparse
import os

import datasets  # pylint: disable=unused-import
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet

MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/pose/models/"


def parse_args() -> argparse.Namespace:
    """Arguements definitions."""
    parser = argparse.ArgumentParser(description="mmpose test model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-dir", help="directory where the outputs are saved."
    )
    parser.add_argument(
        "--gpu_collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. For example, "
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    """Main function for model inference."""
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg.load_from = MODEL_SERVER + cfg_name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
        dist=distributed,
        shuffle=False,
        drop_last=False,
    )
    dataloader_setting = dict(
        dataloader_setting, **cfg.data.get("test_dataloader", {})
    )
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, cfg.load_from, map_location="cpu")

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )

    rank, _ = get_dist_info()

    if rank == 0:
        dataset.convert_format(outputs, args.format_dir)


if __name__ == "__main__":
    main()
