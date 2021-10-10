"""Visualization script for drivable area."""

import argparse
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from bdd100k.common.logger import logger
from bdd100k.common.utils import list_files
from matplotlib.axes import Axes
from PIL import Image
from scalabel.common.parallel import NPROC
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="masks/bitmasks to colormaps")
    parser.add_argument("-i", "--image", help="path to images.")
    parser.add_argument("-c", "--color", help="path to colorized bitmasks.")
    parser.add_argument(
        "-o", "--output", help="path to save generated colormaps."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=-1,
        help="number of samples to generate, set to -1 for all samples.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=NPROC,
        help="number of processes for conversion.",
    )
    return parser.parse_args()


def vis_mask(image_file: str, colormap_file: str, out_path: str) -> None:
    """Visualize bitmask for one image."""
    img = np.array(Image.open(image_file))
    bitmask = np.array(Image.open(colormap_file).convert("RGB"))
    figsize = (int(1280 // 80), int(720 // 80))
    fig = plt.figure(figsize=figsize, dpi=80)
    ax: Axes = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    ax.axis("off")
    ax.imshow(img, interpolation="bilinear", aspect="auto")
    # masking out background pixels
    mask = np.repeat((bitmask.sum(axis=2) > 0)[:, :, np.newaxis], 3, axis=2)
    ax.imshow(np.where(mask, bitmask, img), alpha=0.5)
    plt.savefig(out_path, dpi=80)
    plt.close()


def vis_masks(
    image_files: List[str],
    colormap_files: List[str],
    out_paths: List[str],
    nproc: int = NPROC,
) -> None:
    """Visualize bitmasks for a list of images."""
    logger.info("Visualizing bitmasks...")

    with Pool(nproc) as pool:
        pool.starmap(
            partial(vis_mask),
            tqdm(
                zip(image_files, colormap_files, out_paths),
                total=len(image_files),
            ),
        )


def vis(
    image_dir: str,
    color_dir: str,
    out_base: str,
    num_samples: int,
    nproc: int = NPROC,
) -> None:
    """Visualize drivable area bitmasks."""
    files = list_files(image_dir, ".jpg")
    image_files: List[str] = []
    colormap_files: List[str] = []
    out_paths: List[str] = []

    logger.info("Preparing bitmasks for visualization")

    if num_samples >= 0:
        files = files[:num_samples]
    for file_name in tqdm(files):
        image_path = os.path.join(image_dir, file_name)
        color_path = os.path.join(color_dir, file_name.replace(".jpg", ".png"))
        out_path = os.path.join(out_base, file_name.replace(".jpg", ".png"))
        image_files.append(image_path)
        colormap_files.append(color_path)
        out_paths.append(out_path)
    os.makedirs(os.path.dirname(out_paths[0]), exist_ok=True)
    vis_masks(image_files, colormap_files, out_paths, nproc)


def main() -> None:
    """Main function."""
    args = parse_args()
    vis(args.image, args.color, args.output, args.num_samples, args.nproc)


if __name__ == "__main__":
    main()
