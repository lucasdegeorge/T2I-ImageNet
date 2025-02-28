from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Union, Tuple
import os
from PIL import Image
from pathlib import Path
import torch


class CaptioningDataset(Dataset):
    def __init__(
        self,
        image_dir: Union[str, Path],
        batch_size: int = 32,
        transform: Union[bool, transforms.Compose] = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        assert image_dir, "Image directory must be provided."
        self.image_dir = Path(image_dir)
        self.image_paths = [
            Path(path)
            for path in list_files(image_dir)
            if path.endswith((".JPEG", ".jpg", ".png"))
        ]
        print(f"Number of images in {image_dir} folder: {len(self.image_paths)}")
        self.batch_size = batch_size
        self.device = device

        if isinstance(transform, bool):
            if transform:
                self.transform = transforms.Compose(
                    [  ## Similar to HF's PaliGemma processor with defaults parameters
                        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
            else:
                self.transform = None
        else:
            self.transform = transform

    def __len__(self) -> int:
        if self.batch_size == 1:
            return len(self.image_paths)
        elif self.batch_size > 1 and self.batch_size < len(self.image_paths):
            if len(self.image_paths) % self.batch_size == 0:
                return len(self.image_paths) // self.batch_size
            else:
                return len(self.image_paths) // self.batch_size + 1
        else:
            raise ValueError(
                f"Batch size {self.batch_size} must be greater than 1 and less than the total number of images {len(self.image_paths)}."
            )

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[Image.Image, str], Tuple[list, list]]:
        if idx >= len(self):
            raise IndexError("Index out of range")
        else:
            if self.batch_size == 1:
                return self.get_1_item(idx)
            elif self.batch_size > 1:
                return self.get_batch(idx)

    def get_1_item(self, idx: int) -> Tuple[Image.Image, str]:
        """Called by __getitem__. Do not call directly."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

    def get_batch(self, idx: int) -> Tuple[list, list]:
        """Called by __getitem__. Do not call directly."""
        images = []
        img_paths = []
        for i in range(
            idx * self.batch_size,
            min((idx + 1) * self.batch_size, len(self.image_paths)),
        ):
            img_path = self.image_paths[i]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            img_paths.append(img_path)
        return images, img_paths
    

def list_files(directory):
    """Get a list of all files in a directory and its subdirectories."""
    files_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            files_list.append(os.path.join(root, file))
    return files_list