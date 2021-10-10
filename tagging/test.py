"""Inference a pretrained model."""

import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn(
        "wrap_fp16_model from mmcls will be deprecated."
        "Please install mmcv>=1.1.4."
    )
    assert False

import datasets  # pylint: disable=unused-import
import models  # pylint: disable=unused-import, import-error

MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/tagging/models/"


def parse_args() -> argparse.Namespace:
    """Arguements definitions."""
    parser = argparse.ArgumentParser(description="mmcls test model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--out", help="output result directory")
    parser.add_argument(
        "--gpu_collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument(
        "--metric-options",
        nargs="+",
        action=DictAction,
        default={},
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be parsed as a dict metric_options for dataset.evaluate()"
        " function.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device used for testing",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    """Main function for model inference."""
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.load_from is None:
        cfg_split = args.config.split("/")
        cfg_name = f"{cfg_split[-2]}/{cfg_split[-1].replace('.py', '.pth')}"
        cfg.load_from = MODEL_SERVER + cfg_name
    if args.options is not None:
        cfg.merge_from_dict(args.options)
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
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True,
    )

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, cfg.load_from, map_location="cpu")

    if "CLASSES" in checkpoint.get("meta", {}):
        classes = checkpoint["meta"]["CLASSES"]
    else:
        warnings.simplefilter("once")
        warnings.warn(
            "Class names are not saved in the checkpoint's "
            "meta data, exiting."
        )
        assert False

    if not distributed:
        if args.device == "cpu":
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = classes
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
        results = {}
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        eval_results = dataset.evaluate(outputs, metrics, args.metric_options)
        results.update(eval_results)
        for k, v in eval_results.items():
            print(f"\n{k} : {v:.2f}")

        if args.out:
            print(f"\ndumping results to {args.out}")
            os.makedirs(args.out, exist_ok=True)
            mmcv.dump(results, os.path.join(args.out, "results.json"))
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [classes[lb] for lb in pred_label]
            preds = {
                "class_scores": scores,
                "pred_score": pred_score,
                "pred_label": pred_label,
                "pred_class": pred_class,
            }
            mmcv.dump(preds, os.path.join(args.out, "preds.json"))


if __name__ == "__main__":
    main()
