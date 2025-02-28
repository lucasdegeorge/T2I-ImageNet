import random
import string
from copy import deepcopy
import os

import torch
import torch.nn.functional as F
import transformers
import numpy as np
import PIL
import webdataset as wds
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
from pathlib import Path
from utils.image_processing import remap_image_torch

from .classifier_acc import ClassifierAccuracy
from .inception_metrics import MultiInceptionMetrics, NoTrainInceptionV3


class SampleAndEval:
    def __init__(
        self,
        logger,
        root_dir,
        num_images=50000,
        compute_conditional=True,
        compute_unconditional=False,
        compute_per_class_metrics=True,
        log_prefix="val",
        num_classes=10,
        eval_set="val",
        dataset_name="CIFAR-10",
        dataset_type="class_conditional",
        cfg_rate=0.0,
        shape=(4, 32, 32),
        negative_prompts=None,
        coherence_value=1.0,
        networks=["inception"],
    ):
        features = {}
        if "clip" in networks:
            features["clip"] = CLIPFeatureExtractor(
                path="openai/clip-vit-large-patch14"
            )
        if "dinov2" in networks:
            features["dinov2"] = Dinov2FeatureExtractor()
        if "inception" in networks:
            features["inception"] = NoTrainInceptionV3(
                name="inception-v3-compat",
                features_list=[str(2048), "logits_unbiased"],
            )
        if features == {}:
            raise ValueError("No features to compute")
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            features,
            reset_real_features=False,
            compute_unconditional_metrics=compute_unconditional,
            compute_conditional_metrics=compute_conditional,
            compute_conditional_metrics_per_class=compute_per_class_metrics,
            num_classes=num_classes,
            num_inception_chunks=10,
            manifold_k=3,
        )
        if dataset_type == "class_conditional":
            self.classifier_acc = ClassifierAccuracy(dataset=dataset_name)
        elif dataset_type == "text_conditional":
            self.clip_score = CLIPScore("openai/clip-vit-large-patch14")
        else:
            raise ValueError(f"Unknown dataset_type {dataset_type}")

        self.dataset_type = dataset_type
        self.logger = logger
        self.num_images = num_images
        self.compute_conditional = compute_conditional
        self.compute_unconditional = compute_unconditional
        self.log_prefix = log_prefix
        self.true_features_computed = False
        self.eval_set = eval_set
        self.num_classes = num_classes
        self.cfg_rate = cfg_rate
        self.shape = shape
        self.negative_prompts = negative_prompts
        self.coherence_value = coherence_value
        self.root_dir = root_dir

    def compute_and_log_metrics(self, device, pl_module, datamodule, step=0, epoch=0):
        with torch.no_grad():
            self.inception_metrics.to(device)
            if self.dataset_type == "class_conditional":
                self.classifier_acc.to(device)
            elif self.dataset_type == "text_conditional":
                self.clip_score.to(device)
            if self.eval_set == "val":
                dataloader = datamodule.val_dataloader()
            elif self.eval_set == "train":
                dataloader = datamodule.train_dataloader()
            else:
                raise ValueError(f"Unknown eval_set {self.eval_set}")
            dataloader = deepcopy(dataloader)
            if self.dataset_type == "class_conditional":
                dataloader.dataset.transform = datamodule.val_aug
            pl_module = pl_module.to(device)
            pl_module.eval()
            if (
                not self.true_features_computed
                or not self.inception_metrics.reset_real_features
            ):
                self.compute_true_images_features(dataloader)
                self.true_features_computed = True
            if self.compute_unconditional:
                self.compute_fake_images_features(
                    pl_module, dataloader, conditional=False
                )
            if self.compute_conditional:
                self.compute_fake_images_features(
                    pl_module, dataloader, conditional=True
                )

            metrics = self.inception_metrics.compute()
            metrics = {f"{self.log_prefix}/{k}": v for k, v in metrics.items()}
            metrics["trainer/global_step"] = step
            metrics["epoch"] = epoch
            if self.compute_conditional and self.dataset_type == "class_conditional":
                metrics[
                    f"{self.log_prefix}/classifier_acc"
                ] = self.classifier_acc.compute()
            elif self.compute_conditional and self.dataset_type == "text_conditional":
                metrics[f"{self.log_prefix}/clip_score"] = self.clip_score.compute()
            print(metrics)
            self.logger.experiment.log(
                metrics,
            )

    def compute_true_images_features(self, dataloader):
        if isinstance(dataloader, wds.DataPipeline):
            dataset_length = dataloader.pipeline[0].dataset.num_samples
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            dataset_length = len(dataloader.dataset)
        else:
            raise ValueError(f"Unknown dataloader type {type(dataloader)}")
        if dataset_length < self.num_images:
            max_images = dataset_length
        else:
            max_images = self.num_images

        print("Computing true images features")
        max_batch_size = 0
        for i, batch in enumerate(tqdm(dataloader)):
            images = batch["image"]
            cond = batch["label"] if "label" in batch else None
            max_batch_size = max(max_batch_size, images.shape[0])
            if i * max_batch_size >= max_images:
                break
            elif i * max_batch_size + images.shape[0] > max_images:
                images = images[: max_images - i * max_batch_size]
                cond = cond[: max_images - i * max_batch_size]
            self.inception_metrics.update(
                remap_image_torch(images).to(self.inception_metrics.device),
                cond,
                image_type="real",
            )

    def compute_fake_images_features(self, pl_module, dataloader, conditional=False):
        if isinstance(dataloader, wds.DataPipeline):
            dataset_length = dataloader.pipeline[0].dataset.num_samples
        elif isinstance(dataloader, torch.utils.data.DataLoader):
            dataset_length = len(dataloader.dataset)
        else:
            raise ValueError(f"Unknown dataloader type {type(dataloader)}")
        if dataset_length < self.num_images:
            max_images = dataset_length
        else:
            max_images = self.num_images

        print("Computing fake images features")
        max_batch_size = 0
        for i, batch in enumerate(tqdm(dataloader)):
            if self.dataset_type == "text_conditional":
                images = batch["image"]
                if f"{pl_module.cfg.text_embedding_name}_embeddings" in batch:
                    cond = batch
                    text = batch["text"]
                    cond = {
                        k: v.to(pl_module.device)
                        for k, v in cond.items()
                        if type(v) == torch.Tensor
                    }
                    if self.negative_prompts == "random_prompt":
                        unconfident_prompt = torch.from_numpy(
                            np.load(
                                Path(self.root_dir)
                                / Path("cad/utils/flan_t5_xl_random.npy")
                            )
                        )
                        unconfident_prompt = unconfident_prompt.to(pl_module.device)
                    else:
                        unconfident_prompt = None
                else:
                    text = batch["text"]
                    cond = text
                    if self.negative_prompts == "random_prompt":
                        unconfident_prompt = generate_negative_prompts(batch_size)
                    else:
                        unconfident_prompt = None
            else:
                images = batch["image"]
                cond = batch["label"] if "label" in batch else None
            shape = self.shape
            batch_size = images.shape[0]
            max_batch_size = max(max_batch_size, batch_size)
            if i * max_batch_size >= max_images:
                break
            elif i * max_batch_size + batch_size > max_images:
                batch_size = max_images - i * max_batch_size
                cond = cond[:batch_size]
            if isinstance(cond, torch.Tensor):
                cond = cond.to(pl_module.device)
            with torch.no_grad():
                if conditional:
                    images = pl_module.sample(
                        batch_size,
                        shape,
                        cond=cond,
                        stage=self.log_prefix,
                        cfg=self.cfg_rate,
                        unconfident_prompt=unconfident_prompt,
                        coherence_value=self.coherence_value,
                    )
                    self.inception_metrics.update(
                        images, cond, image_type="conditional"
                    )
                    if self.dataset_type == "class_conditional":
                        self.classifier_acc.update(images, cond)
                    elif self.dataset_type == "text_conditional":
                        self.clip_score.update(images, text)
                else:
                    images_uncond = pl_module.sample(
                        batch_size,
                        shape,
                        stage=self.log_prefix,
                        cfg=self.cfg_rate,
                        unconfident_prompt=unconfident_prompt,
                        coherence_value=self.coherence_value,
                    )
                    self.inception_metrics.update(
                        images_uncond, image_type="unconditional"
                    )


def generate_random_string():
    words = []
    for _ in range(random.randint(3, 6)):
        word = "".join(
            random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 7))
        )
        words.append(word)
    return " ".join(words)


def generate_negative_prompts(batch_size):
    return [generate_random_string() for _ in range(batch_size)]


class CLIPFeatureExtractor(torch.nn.Module):
    def __init__(self, path="openai/clip-vit-large-patch14"):
        super().__init__()
        self.processor = transformers.CLIPImageProcessor.from_pretrained(path)
        self.model = transformers.CLIPVisionModel.from_pretrained(path)
        self.model.eval().requires_grad_(False)

    def forward(self, images):
        images = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        features = self.model(**images).last_hidden_state[:, 0, :]
        return features, torch.randn(
            features.shape[0], 1000, dtype=torch.float, device=features.device
        )


class Dinov2FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        os.environ["XFORMERS_DISABLED"] = "1"
        self.model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        self.model.eval().requires_grad_(False)

    def forward(self, images):
        device = images.device
        x = images.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        x = np.stack(
            [
                np.uint8(
                    PIL.Image.fromarray(xx, "RGB").resize(
                        (224, 224), PIL.Image.Resampling.BICUBIC
                    )
                )
                for xx in x
            ]
        )
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        x = x.to(torch.float32) / 255
        x = x - torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, -1, 1, 1)
        x = x / torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(
            1, -1, 1, 1
        )
        features = self.model(x)
        return features, torch.randn(
            features.shape[0], 1000, dtype=torch.float, device=features.device
        )
