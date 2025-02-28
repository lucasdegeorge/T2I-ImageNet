import math

import pytorch_lightning as L
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class ImageDataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        full_batch_size,
        num_workers,
        collate_fn=default_collate,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.num_workers = num_workers
        self._train_dataset_builder = train_dataset
        self._val_dataset_builder = val_dataset
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self._train_dataset_builder()
        self.val_dataset = self._val_dataset_builder()
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        self.train_aug = self.train_dataset.transform
        self.val_aug = self.val_dataset.transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            # collate_fn=collate_to_dict(["image", "label"]),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class WebdatasetDataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        full_batch_size,
        num_workers,
        collate_fn=default_collate,
        num_nodes=1,
        num_devices=1,
    ):
        super().__init__()
        num_devices = num_devices if type(num_devices) == int else len(num_devices)
        self.full_batch_size = full_batch_size
        self.batch_size = full_batch_size // (num_nodes * num_devices)
        print(f"Each GPU will receive {self.batch_size} images")
        self.num_workers = num_workers
        self.world_size = num_nodes * num_devices
        self._train_dataset_builder = train_dataset
        self._val_dataset_builder = val_dataset
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self._train_dataset_builder()
        self.val_dataset = self._val_dataset_builder()
        self.train_dataset = self.train_dataset.compose(
            wds.batched(
                self.batch_size,
                partial=self.world_size > 1,
                collation_fn=self.collate_fn,
                # dict_collate_and_pad(["flan_t5_xl"], max_length=256),
            )
        )
        num_train_samples = self.train_dataset.num_samples
        if self.world_size > 1:
            self.num_train_batches = math.ceil(num_train_samples / self.full_batch_size)
            num_workers = max(1, self.num_workers)

            num_train_worker_batches = math.ceil(self.num_train_batches / num_workers)
            self.num_train_batches = num_train_worker_batches * num_workers
            num_train_samples = self.num_train_batches * self.full_batch_size

            self.train_dataset = self.train_dataset.with_epoch(
                num_train_worker_batches
            ).with_length(num_train_worker_batches)
        else:
            self.num_train_batches = math.ceil(num_train_samples / self.batch_size)

            self.train_dataset = self.train_dataset.with_epoch(
                self.num_train_batches
            ).with_length(self.num_train_batches)
        self.train_aug = self.train_dataset.image_transforms
        self.val_aug = self.val_dataset.image_transforms

    def train_dataloader(self):
        return wds.WebLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=self.num_workers > 1,
        ).with_length(self.num_train_batches)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


def dict_collate_and_pad(keys_to_pad, max_length):
    def dict_collate(batch):
        output_dict = {}
        if isinstance(batch[0], dict):
            for key in batch[0].keys():
                list_key = [d[key] for d in batch]
                if key not in keys_to_pad:
                    output_dict[key] = dict_collate(list_key)
                else:
                    output_dict[f"{key}_embeddings"] = torch.zeros(
                        len(list_key),
                        max_length,
                        list_key[0].shape[-1],
                        dtype=list_key[0].dtype,
                    )
                    output_dict[f"{key}_mask"] = torch.zeros(
                        len(list_key), max_length, dtype=torch.bool
                    )
                    for i, x in enumerate(list_key):
                        output_dict[f"{key}_embeddings"][
                            i, : min(len(x), max_length)
                        ] = x[:max_length]
                        output_dict[f"{key}_mask"][i, : min(len(x), max_length)] = 1
        else:
            return default_collate(batch)
        return output_dict

    return dict_collate


def collate_to_dict(keys):
    def collate(batch):
        output_dict = {}
        collated_batch = default_collate(batch)
        for i, key in enumerate(keys):
            output_dict[key] = collated_batch[i]
        return output_dict

    return collate
