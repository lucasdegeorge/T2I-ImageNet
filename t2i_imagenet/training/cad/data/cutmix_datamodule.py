import math

import pytorch_lightning as L
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class WebdatasetCutMixDataModule(L.LightningDataModule):
    """
    Module to load image data
    """

    def __init__(
        self,
        train_dataset,
        cutmix_dataset,
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
        self._cutmix_dataset_builder = cutmix_dataset
        self._val_dataset_builder = val_dataset
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.train_dataset = self._train_dataset_builder()
        self.cutmix_dataset = self._cutmix_dataset_builder()
        self.val_dataset = self._val_dataset_builder()
        self.train_dataset = self.train_dataset.compose(
            wds.batched(
                self.batch_size,
                partial=self.world_size > 1,
                collation_fn=self.collate_fn,
                # dict_collate_and_pad(["flan_t5_xl"], max_length=256),
            )
        )
        self.cutmix_dataset = self.cutmix_dataset.compose(
            wds.batched(
                self.batch_size,
                partial=self.world_size > 1,
                collation_fn=self.collate_fn,
                # dict_collate_and_pad(["flan_t5_xl"], max_length=256),
            )
        )
        num_train_samples = self.train_dataset.num_samples
        num_cutmix_samples = self.cutmix_dataset.num_samples
        if self.world_size > 1:
            self.num_train_batches = math.ceil(num_train_samples / self.full_batch_size)
            self.num_cutmix_batches = math.ceil(num_cutmix_samples / self.full_batch_size)
            num_workers = max(1, self.num_workers)

            num_train_worker_batches = math.ceil(self.num_train_batches / num_workers)
            num_cutmix_worker_batches = math.ceil(self.num_cutmix_batches / num_workers)
            self.num_train_batches = num_train_worker_batches * num_workers
            self.num_cutmix_batches = num_cutmix_worker_batches * num_workers
            num_train_samples = self.num_train_batches * self.full_batch_size
            num_cutmix_samples = self.num_cutmix_batches * self.full_batch_size

            self.train_dataset = self.train_dataset.with_epoch(
                num_train_worker_batches
            ).with_length(num_train_worker_batches)
            self.cutmix_dataset = self.cutmix_dataset.with_epoch(
                num_cutmix_worker_batches
            ).with_length(num_cutmix_worker_batches)
        else:
            self.num_train_batches = math.ceil(num_train_samples / self.batch_size)
            self.num_cutmix_batches = math.ceil(num_cutmix_samples / self.batch_size)

            self.train_dataset = self.train_dataset.with_epoch(
                self.num_train_batches
            ).with_length(self.num_train_batches)
            self.cutmix_dataset = self.cutmix_dataset.with_epoch(
                self.num_cutmix_batches
            ).with_length(self.num_cutmix_batches)

        self.train_aug = self.train_dataset.image_transforms
        self.cutmix_aug = self.cutmix_dataset.image_transforms
        self.val_aug = self.val_dataset.image_transforms

    def train_dataloader(self):
        return [
            ## Train raw images 
            wds.WebLoader(
                self.train_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=self.num_workers,
                # persistent_workers=self.num_workers > 1,
            ).with_length(self.num_train_batches), 

            ## Train cutmix images
            wds.WebLoader(
                self.cutmix_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=self.num_workers,
                # persistent_workers=self.num_workers > 1,
            ).with_length(self.num_cutmix_batches), 
        ]

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

