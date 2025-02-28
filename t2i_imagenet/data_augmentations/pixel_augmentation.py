"""Implements the cutmix image augmentations."""

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import random
from PIL import Image
from torchvision.utils import save_image
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import os

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)


def check_overlap(
    rect1_x1, rect1_y1, rect1_x2, rect1_y2, rect2_x1, rect2_y1, rect2_x2, rect2_y2
):
    """Checks if two rectangles overlap."""
    if rect1_x1 > rect2_x2 or rect2_x1 > rect1_x2:
        return False
    if rect1_y1 > rect2_y2 or rect2_y1 > rect1_y2:
        return False
    return True


def prepare_images(pil_base_image, pil_cutmix_image, base_image_size=256, n=5):
    """Prepares the cutmix images.

    :param pil_base_image: PIL image of the base image.
    :param pil_cutmix_image: PIL image of the cutmix image.
    :param base_image_size: Size of the base image. Default: 256.
    :param n: Setting for the cutmix image. Default: 5.
    """

    ## Augmentations ##
    ## 1. For the base image ##
    base_transform = transforms.Compose(
        [
            transforms.Resize(base_image_size),
            transforms.CenterCrop(base_image_size),
            transforms.ToTensor(),
        ]
    )

    ## 2. For the cutmix image ##

    if n == 5:
        n = random.randint(1, 4)

    ## Scale = same as base --> Setting 1 ##
    if n == 1:
        cutmix_transform_list = [
            transforms.Resize(base_image_size),
            transforms.CenterCrop(base_image_size),
            transforms.ToTensor(),
        ]

        m = random.randint(1, 2)
        if m == 1:
            mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
            x_start = base_image_size // 2
            y_start = 0
            x_end = base_image_size
            y_end = base_image_size

        else:
            mask = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            x_start = 0
            y_start = base_image_size // 2
            x_end = base_image_size
            y_end = base_image_size

        ## Upscaling the mask ##
        mask = F.upsample(
            mask.unsqueeze(0).unsqueeze(0),
            size=(base_image_size, base_image_size),
            mode="nearest",
        )

    ## Scale = half and placed at corners --> Setting 2 ##
    elif n == 2:
        cutmix_image_size = base_image_size // 2
        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
        ]

        m = random.randint(1, 4)
        if m == 1:
            mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
            x_start = 0
            y_start = 0
            x_end = cutmix_image_size
            y_end = cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad((0, 0, cutmix_image_size, cutmix_image_size), fill=0)
            )

        elif m == 2:
            mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
            x_start = base_image_size - cutmix_image_size
            y_start = 0
            x_end = base_image_size
            y_end = cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad((cutmix_image_size, 0, 0, cutmix_image_size), fill=0)
            )

        elif m == 3:
            mask = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
            x_start = 0
            y_start = base_image_size - cutmix_image_size
            x_end = cutmix_image_size
            y_end = base_image_size
            cutmix_transform_list.append(
                transforms.Pad((0, cutmix_image_size, cutmix_image_size, 0), fill=0)
            )

        else:
            mask = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
            x_start = base_image_size - cutmix_image_size
            y_start = base_image_size - cutmix_image_size
            x_end = base_image_size
            y_end = base_image_size
            cutmix_transform_list.append(
                transforms.Pad((cutmix_image_size, cutmix_image_size, 0, 0), fill=0)
            )

        ## Upscaling the mask ##
        mask = F.upsample(
            mask.unsqueeze(0).unsqueeze(0),
            size=(base_image_size, base_image_size),
            mode="nearest",
        )

    ## Scale = 1/3 and placed along the borders --> Setting 3 ##
    elif n == 3:
        # cutmix_image_size = base_image_size // 4
        cutmix_image_size = base_image_size // 3
        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
        ]

        mask = torch.zeros(base_image_size, base_image_size)

        m = random.randint(1, 4)

        if m == 1:
            x_start = 0
            y_start = random.randint(0, base_image_size - cutmix_image_size)
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        0,
                        y_start,
                        base_image_size - cutmix_image_size,
                        base_image_size - (y_start + cutmix_image_size),
                    ),
                    fill=0,
                )
            )

        elif m == 2:
            x_start = random.randint(0, base_image_size - cutmix_image_size)
            y_start = 0
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        0,
                        base_image_size - (x_start + cutmix_image_size),
                        base_image_size - cutmix_image_size,
                    ),
                    fill=0,
                )
            )

        elif m == 3:
            x_start = base_image_size - cutmix_image_size
            y_start = random.randint(0, base_image_size - cutmix_image_size)
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        y_start,
                        0,
                        base_image_size - (y_start + cutmix_image_size),
                    ),
                    fill=0,
                )
            )

        else:
            x_start = random.randint(0, base_image_size - cutmix_image_size)
            y_start = base_image_size - cutmix_image_size
            cutmix_transform_list.append(
                transforms.Pad(
                    (
                        x_start,
                        y_start,
                        base_image_size - (x_start + cutmix_image_size),
                        0,
                    ),
                    fill=0,
                )
            )

        mask[
            y_start : y_start + cutmix_image_size, x_start : x_start + cutmix_image_size
        ] = 1.0
        x_end = x_start + cutmix_image_size
        y_end = y_start + cutmix_image_size

        mask = mask.unsqueeze(0).unsqueeze(0)

    ## Scale quarter but not in the border and not in the center --> Setting 4 ##
    else:
        center = base_image_size // 2
        red_zone_x_start = center - int(0.1 * base_image_size)
        red_zone_x_end = center + int(0.1 * base_image_size)
        red_zone_y_start = center - int(0.1 * base_image_size)
        red_zone_y_end = center + int(0.1 * base_image_size)

        cutmix_image_size = base_image_size // 4

        val = True
        while val:
            x_start = random.randint(
                int(0.05 * base_image_size),
                base_image_size - cutmix_image_size - int(0.05 * base_image_size),
            )
            y_start = random.randint(
                int(0.05 * base_image_size),
                base_image_size - cutmix_image_size - int(0.05 * base_image_size),
            )
            x_end = x_start + cutmix_image_size
            y_end = y_start + cutmix_image_size

            if check_overlap(
                red_zone_x_start,
                red_zone_y_start,
                red_zone_x_end,
                red_zone_y_end,
                x_start,
                y_start,
                x_end,
                y_end,
            ):
                continue

            else:
                val = False

        # print(red_zone_x_start, red_zone_x_end, red_zone_y_start, red_zone_y_end)
        # print(x_start, x_end, y_start, y_end)

        cutmix_transform_list = [
            transforms.Resize(cutmix_image_size),
            transforms.CenterCrop(cutmix_image_size),
            transforms.ToTensor(),
            transforms.Pad(
                (
                    x_start,
                    y_start,
                    base_image_size - x_end,
                    base_image_size - y_end,
                ),
                fill=0,
            ),
        ]

        mask = torch.zeros(base_image_size, base_image_size)
        mask[
            y_start : y_start + cutmix_image_size, x_start : x_start + cutmix_image_size
        ] = 1.0

        mask = mask.unsqueeze(0).unsqueeze(0)

    ## Creating the final cutmix transform ##
    cutmix_transform = transforms.Compose(cutmix_transform_list)
    ## Apply the transformations ##
    base_image = base_transform(pil_base_image)
    cutmix_image = cutmix_transform(pil_cutmix_image)

    # print(f"Scale : {n}")
    final_img = base_image * (1 - mask.to(base_image.device)) + cutmix_image * mask.to(
        base_image.device
    )

    return final_img, x_start, y_start, x_end, y_end


def create_all_images(
    img_dir, base_image_size, setting, dataset_times, out_dir, img_extension
):
    """Creates all the cutmix images.

    :param img_dir: Directory of the images.
    :param base_image_size: Size of the base image.
    :param setting: Setting for the cutmix image.
    :param dataset_times: How many times of imagenet is the cutmix dataset?
    :param out_dir: Directory to save the images.
    :param img_extension: Extension of the images.
    """
    ## Getting all the directories of imagenet ##
    class_directories = sorted(glob(f"{img_dir}/*"))

    ## Looping over all class directories ##
    for idx, class_directory in enumerate(class_directories):
        ## Getting all the image paths of the current class ##
        current_class_image_paths = sorted(glob(f"{class_directory}/*.{img_extension}"))

        ## Other class directories ##
        other_class_directories = class_directories.copy()
        other_class_directories.remove(class_directory)

        ## Getting all the image paths of the other classes ##
        other_class_image_paths = []
        for class_directory in class_directories:
            other_class_image_paths.extend(glob(f"{class_directory}/*.{img_extension}"))

        ## Looping over the images of the current class ##
        for idx, img_path in tqdm(
            enumerate(current_class_image_paths), total=len(current_class_image_paths)
        ):
            ## Getting the image name. Used for saving. ##
            base_img_name = os.path.basename(img_path).split(".")[0]

            ## Loading the image ##
            base_img = Image.open(img_path).convert("RGB")

            ## Getting random images from the other classes ##
            cutmix_img_paths = random.sample(other_class_image_paths, dataset_times)

            ## Looping over the cutmix images ##
            for cutmix_img_path in cutmix_img_paths:
                ## Loading the random image ##
                cutmix_img = Image.open(cutmix_img_path).convert("RGB")

                ## Getting the image name ##
                cutmix_img_name = os.path.basename(cutmix_img_path).split(".")[0]

                ## Preparing the images ##
                final_img, x_start, y_start, x_end, y_end = prepare_images(
                    base_img, cutmix_img, base_image_size, setting
                )

                ## Saving the image ##
                save_image(
                    final_img,
                    f"{out_dir}/{base_img_name}_{cutmix_img_name}_{x_start}_{y_start}_{x_end}_{y_end}.{img_extension}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_image_size", type=int, default=256)
    parser.add_argument("--setting", type=int, default=5)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset_times",
        type=int,
        default=1,
        help="How many times of imagenet is the cutmix dataset?",
    )
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--img_extension", type=str, default="JPEG")
    args = parser.parse_args()

    create_all_images(
        args.img_dir,
        args.base_image_size,
        args.setting,
        args.dataset_times,
        args.out_dir,
        args.img_extension,
    )
