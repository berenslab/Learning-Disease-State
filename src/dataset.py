# src/dataset.py
import os
from typing import Tuple, Callable
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from PIL import Image, ImageOps
import random
import cv2
import re


class pad_image_to_target_size(nn.Module):
    def __init__(self, target_width: int = 1024, target_height: int = 496):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height

    def forward(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        padding_left = (self.target_width - width) // 2
        padding_right = self.target_width - width - padding_left
        padding_top = (self.target_height - height) // 2
        padding_bottom = self.target_height - height - padding_top
        return ImageOps.expand(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)
    
class MarioTrainTransforms(object):
    def __init__(self, image_size: Tuple[int], crop_size: Tuple[int]) -> None:
        self.image_transform = torchvision.transforms.Compose(
            [
                pad_image_to_target_size(),
                torchvision.transforms.Resize(image_size, antialias=True),
                torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            ]
        )
    def __call__(self, image):
        return self.image_transform(image)

class MarioTestTransforms(object):
    def __init__(self, image_size: Tuple[int], crop_size: Tuple[int]) -> None:
        self.image_transform = torchvision.transforms.Compose(
            [
                pad_image_to_target_size(),
                torchvision.transforms.Resize(crop_size, antialias=True),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            ]
        )
    def __call__(self, image):
        return self.image_transform(image)

class MarioDatasetTask1(torch.utils.data.Dataset):
    """Pytorch dataset for MARIO Challenge, first task.

    https://youvenz.github.io/MARIO_challenge.github.io

    TASK 1: Classify AMD evolution between two pairs of 2-D slices from two consecutive
    2D OCT acquisitions.
    Labels: 0: reduced (eliminated or persistent reduced)
            1: stable (inactive or persistent stable)
            2: worsened (persistent worsened or appeared)
            3: other (uninterpretable or appeared and eliminated)

    Attributes:
        root: Root directory of dataset.
        split: One of `train`, `val`, `test`.
        image_transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        valtest: bool = False,
        image_size = (256,440),
        crop_size = (224,385),
    ):
        self._meta = pd.read_csv(
            os.path.join(root, f"df_task1_{split}_challenge.csv"),
        )

        self.valtest = valtest

        if valtest:
            self.image_transform = MarioTestTransforms(image_size, crop_size)
        else:
            self.image_transform = MarioTrainTransforms(image_size, crop_size)

        self._labels = torch.tensor(self._meta["label"].to_list(), dtype=torch.int8)
        self._image_paths_t = self._meta["image_at_ti"].to_list()
        self._image_paths_t_next = self._meta["image_at_ti+1"].to_list()
        
        self.root_dir = root
        self.split = split

    def __len__(self) -> int:
        return len(self._image_paths_t)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        image_path_t = os.path.join(
            self.root_dir,
            self.split,
            self._image_paths_t[index],
        )
        image_path_t_next = os.path.join(
            self.root_dir,
            self.split,
            self._image_paths_t_next[index],
        )

        image_t = Image.open(image_path_t)
        image_t_next = Image.open(image_path_t_next)

        rand_seed = torch.randint(0, 100000, (1,)).item()
        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        image_t = self.image_transform(image_t)

        torch.manual_seed(rand_seed)
        random.seed(rand_seed)
        image_t_next = self.image_transform(image_t_next)

        index = torch.tensor(index, dtype=torch.int64)

        label = self._labels[index]
        
        return image_t, image_t_next, label, index