from datasets import load_dataset
import os
from PIL import Image
import numpy as np
import torch
import albumentations.augmentations.transforms as T
import torchvision.transforms as TT
import albumentations as A
import cv2
from glob import glob


def add_low_light_noise(image, **kwargs):
    """
    Adds random realistic noise to an image to simulate low-light conditions.
    Expects 'color_shift_range' and 'noise_dispersion_range' to be passed in 'kwargs'.
    """
    color_shift_range = kwargs.get("color_shift_range", (1, 3))  # 5 10
    noise_dispersion_range = kwargs.get(
        "noise_dispersion_range", (0.01, 0.05)
    )  # .02 .09

    # Sampling of the intensity of color noise and of the dispersion of the Gaussian noise
    color_shift = np.random.uniform(*color_shift_range)
    noise_dispersion = np.random.uniform(*noise_dispersion_range)

    # Gaussian noise
    h, w, c = image.shape
    gaussian_noise = np.random.normal(0, image.std() * noise_dispersion, (h, w, c))
    # Color noise
    color_noise = np.random.normal(0, color_shift, (h, w, c))

    noisy_image = image + gaussian_noise + color_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


if __name__ == "__main__":
    np.random.seed(42)

    for split in ["train", "test", "val"]:
        os.makedirs(f"data/finetuning/{split}", exist_ok=True)
        os.makedirs(f"data/finetuning/{split}/imgs", exist_ok=True)
        os.makedirs(f"data/finetuning/{split}/targets", exist_ok=True)

    transform = A.Compose(
        [
            T.ColorJitter(
                brightness=(0.1, 0.2),
                contrast=0,
                saturation=0,
                hue=0,
                always_apply=True,
                p=1,
            ),  # Artificial darkening
            A.Lambda(image=add_low_light_noise),  # Randomized low-light gaussian noise
            A.SmallestMaxSize(max_size=400),
        ]
    )

    transform_ground_truth = A.Compose([A.SmallestMaxSize(max_size=400)])

    original_images = glob("data/finetuning/original_images/*")

    # Dataset splitting
    total_size = len(original_images)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : (train_size + val_size)]
    test_indices = indices[(train_size + val_size) :]

    def split_of_image(index):
        if index in train_indices:
            return "train"
        elif index in val_indices:
            return "val"
        else:
            return "test"

    too_small_images = 0
    total_images = 0
    for idx, image_path in enumerate(original_images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        transformed_ground_truth = transform_ground_truth(image=image)["image"]
        try:
            if transformed_ground_truth.shape[0] > transformed_ground_truth.shape[1]:
                transformed_ground_truth = A.CenterCrop(
                    height=704, width=400, always_apply=True, p=1
                )(image=transformed_ground_truth)["image"]
                transformed_image = A.CenterCrop(
                    height=704, width=400, always_apply=True, p=1
                )(image=transformed_image)["image"]
            else:
                transformed_ground_truth = A.CenterCrop(
                    height=400, width=704, always_apply=True, p=1
                )(image=transformed_ground_truth)["image"]
                transformed_image = A.CenterCrop(
                    height=400, width=704, always_apply=True, p=1
                )(image=transformed_image)["image"]

            split = split_of_image(idx)
            cv2.imwrite(
                f"data/finetuning/{split}/imgs/img_{idx}.png",
                cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                f"data/finetuning/{split}/targets/img_{idx}.png",
                cv2.cvtColor(transformed_ground_truth, cv2.COLOR_RGB2BGR),
            )
        except:
            too_small_images += 1
        total_images += 1
    print(
        too_small_images,
        "too small images weren't integrated to the dataset out of a total of ",
        total_images,
        "images",
    )  # Decided to perform sorting in the dataset generation rather than at manual data curation
