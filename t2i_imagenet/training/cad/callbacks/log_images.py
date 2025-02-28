import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import contextlib
import math
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from pytorch_lightning import Callback
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from data.datamodule import dict_collate_and_pad


@contextlib.contextmanager
def temp_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class LogGeneratedImages(Callback):
    def __init__(
        self,
        root_dir,
        mode="class_cond",
        num_samples_per_cond: int = 4,
        num_samples_unconditional: int = 9,
        num_classes=10,
        shape=(3, 32, 32),
        log_every_n_steps: int = 1,
        max_classes=20,
        cfg_rate=0,
        log_conditional=True,
        log_unconditional=False,
        text_embedding_name=None,
        negative_prompts=None,
        batch_size=128,
    ):
        super().__init__()
        self.num_samples_per_cond = num_samples_per_cond
        self.num_samples_unconditional = num_samples_unconditional
        # Assert that num_samples_per_cond is a perfect square
        assert math.sqrt(num_samples_per_cond) % 1 == 0
        assert math.sqrt(num_samples_unconditional) % 1 == 0
        self.shape = shape
        self.sqrt_num_samples_per_cond = int(math.sqrt(num_samples_per_cond))
        self.sqrt_num_samples_unconditional = int(math.sqrt(num_samples_unconditional))
        self.log_conditional = log_conditional
        self.log_unconditional = log_unconditional
        self.mode = mode
        if self.mode == "class_conditional":
            self.num_classes = num_classes
            self.max_classes = max_classes
        elif self.mode == "text_conditional":
            self.text_embedding_name = text_embedding_name
        self.log_every_n_steps = log_every_n_steps
        self.ready = True
        self.batch_size = min(batch_size, 64)
        self.root_dir = root_dir
        self.cfg_rate = cfg_rate
        if negative_prompts == "random_prompt":
            self.negative_prompts = torch.from_numpy(
                np.load(Path(root_dir) / Path("cad/utils/flan_t5_xl_random.npy"))
            )
        else:
            self.negative_prompts = None

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_start(self, trainer, pl_module):
        try:
            self.world_size = torch.distributed.get_world_size()
        except:
            self.world_size = 1
        generator = torch.Generator(device="cpu").manual_seed(3407)
        if self.log_conditional:
            if self.mode == "class_conditional":
                self.train_dataset = ClassCondPromptBed(
                    self.num_classes,
                    self.num_samples_per_cond,
                    max_classes=self.max_classes,
                    generator=generator,
                    shape=self.shape,
                )
            elif self.mode == "text_conditional":
                self.train_dataset = TextCondPromptBed(
                    Path(self.root_dir) / Path("t2i_imagenet/training/cad/datasets/text_prompt_testbed"),
                    self.num_samples_per_cond,
                    self.text_embedding_name,
                    generator=generator,
                    shape=self.shape,
                )
        if self.log_unconditional:
            generator = torch.Generator(device="cpu").manual_seed(3407)
            self.x_N_train_uncond = torch.randn(
                self.num_samples_unconditional,
                *self.shape,
                device=pl_module.device,
                generator=generator,
            )

    def on_test_start(self, trainer, pl_module):
        try:
            self.world_size = torch.distributed.get_world_size()
        except:
            self.world_size = 1
        generator = torch.Generator(device="cpu").manual_seed(3407)
        if self.log_conditional:
            if self.mode == "class_conditional":
                self.test_dataset = ClassCondPromptBed(
                    self.num_classes,
                    self.num_samples_per_cond,
                    max_classes=self.max_classes,
                    generator=generator,
                    shape=self.shape,
                )
            elif self.mode == "text_conditional":
                self.test_dataset = TextCondPromptBed(
                    Path(self.root_dir) / "t2i_imagenet/training/cad/datasets/text_prompt_testbed",
                    self.num_samples_per_cond,
                    self.text_embedding_name,
                    generator=generator,
                    shape=self.shape,
                )
        if self.log_unconditional:
            generator = torch.Generator(device="cpu").manual_seed(3407)
            self.x_N_test_uncond = torch.randn(
                self.num_samples_unconditional,
                *self.shape,
                device=pl_module.device,
                generator=generator,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step % self.log_every_n_steps == 0 and self.ready:
            print("Logging images")
            if self.log_conditional:
                self.log_images(
                    trainer,
                    pl_module,
                    self.train_dataset,
                    prefix="val",
                    conditional=True,
                )
            if self.log_unconditional:
                self.log_images(
                    trainer,
                    pl_module,
                    self.x_N_train_uncond,
                    prefix="val",
                    conditional=False,
                )

    def on_test_end(self, trainer, pl_module):
        if self.log_conditional:
            self.log_images(
                trainer, pl_module, self.test_dataset, prefix="test", conditional=True
            )
        if self.log_unconditional:
            self.log_images(
                trainer,
                pl_module,
                self.x_N_test_uncond,
                prefix="test",
                conditional=False,
            )

    def log_images(self, trainer, pl_module, x_N=None, prefix="val", conditional=True):
        if self.ready:
            logger = trainer.logger
            experiment = logger.experiment
            if (
                isinstance(self.negative_prompts, torch.Tensor)
                and self.negative_prompts.device != pl_module.device
            ):
                self.negative_prompts = self.negative_prompts.to(pl_module.device)
            generator = torch.Generator(pl_module.device).manual_seed(3407)
            if isinstance(x_N, Dataset):
                if self.mode == "class_conditional":
                    collate_fn = torch.utils.data.dataloader.default_collate
                elif self.mode == "text_conditional":
                    collate_fn = dict_collate_and_pad(["flan_t5_xl"], max_length=128)

                if self.world_size > 1:
                    dataloader = DataLoader(
                        x_N,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=4,
                        sampler=DistributedSampler(x_N, shuffle=False, drop_last=False),
                        collate_fn=collate_fn,
                    )
                else:
                    dataloader = DataLoader(
                        x_N,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn,
                    )
            results = {}
            if conditional:
                for batch in tqdm(dataloader):
                    with torch.no_grad():
                        if self.mode == "class_conditional":
                            x_k = batch["x_N"]
                            name = batch["name"]
                            cond = batch["label"]
                            cond = cond.to(pl_module.device)
                            x_k = x_k.to(pl_module.device)
                        elif self.mode == "text_conditional":
                            if len(batch) == 2:
                                x_k = batch["x_N"]
                                cond = batch["text"]
                                x_k = x_k.to(pl_module.device)
                                name = cond
                            elif len(batch) == 4:
                                x_k = batch["x_N"]
                                name = batch["text"]
                                del batch["text"]
                                del batch["x_N"]
                                cond = batch
                                cond = {
                                    k: v.to(pl_module.device) for k, v in cond.items()
                                }
                                x_k = x_k.to(pl_module.device)
                            else:
                                raise ValueError(
                                    f"Unknown batch size {len(batch)} for text conditional"
                                )
                        else:
                            raise ValueError(f"Unknown mode {self.mode}")
                        sampled_images = pl_module.sample(
                            x_N=x_k,
                            cond=cond,
                            stage=prefix,
                            generator=generator,
                            cfg=self.cfg_rate,
                            unconfident_prompt=self.negative_prompts,
                        )
                        if self.world_size > 1:
                            sampled_images = pl_module.all_gather(
                                sampled_images
                            ).flatten(0, 1)
                            name_list = [None for _ in range(self.world_size)]
                            torch.distributed.all_gather_object(name_list, name)
                            name = [item for sublist in name_list for item in sublist]

                    for name, images in zip(name, sampled_images):
                        if name not in results:
                            results[name] = []
                        if len(results[name]) < self.num_samples_per_cond:
                            results[name].append(images)
                logs = {}
                if pl_module.global_rank == 0:
                    for name, images in results.items():
                        images = rearrange(
                            images,
                            "(b1 b2) c h w -> c (b1 h) (b2 w)",
                            b1=self.sqrt_num_samples_per_cond,
                            b2=self.sqrt_num_samples_per_cond,
                        )

                        images = wandb.Image(
                            images.cpu().numpy().transpose(1, 2, 0),
                        )
                        if len(name) > 50:
                            name = name[:50] + "..."
                        if self.mode == "class_conditional":
                            logs[f"{prefix}/Images/Samples class {name}"] = images
                        elif self.mode == "text_conditional":
                            logs[f"{prefix}/Images/Samples text {name}"] = images

                experiment.log(
                    {
                        **logs,
                        "trainer/global_step": pl_module.global_step,
                    }
                )
            else:
                with torch.no_grad():
                    if self.mode == "class_conditional":
                        cond = torch.zeros(
                            x_N.shape[0], self.num_classes, device=x_N.device
                        )
                    elif self.mode == "text_conditional":
                        cond = pl_module.uncond_conditioning
                        if cond == "":
                            cond = ["" for _ in range(x_N.shape[0])]
                        else:
                            cond = cond.repeat(x_N.shape[0], *[1 for _ in cond.shape])
                    x_N = x_N.to(pl_module.device)
                    sampled_images = pl_module.sample(
                        x_N=x_N, cond=None, stage=prefix, generator=generator
                    )
                    formated_images = rearrange(
                        sampled_images,
                        "(b1 b2) c h w -> c (b1 h) (b2 w)",
                        b1=self.sqrt_num_samples_unconditional,
                        b2=self.sqrt_num_samples_unconditional,
                    )
                    # Check for NaNs
                    assert not torch.isnan(formated_images).any()

                formated_images = wandb.Image(
                    formated_images.cpu().numpy().transpose(1, 2, 0),
                )
                experiment.log(
                    {
                        f"{prefix}/Images/Samples uncond": formated_images,
                        "trainer/global_step": pl_module.global_step,
                    }
                )


class ClassCondPromptBed(Dataset):
    def __init__(
        self,
        num_classes,
        num_samples_per_cond,
        max_classes=250,
        generator=None,
        shape=(3, 32, 32),
    ):
        self.num_classes = num_classes
        self.num_samples_per_cond = num_samples_per_cond
        self.num_samples = (
            min(max_classes, self.num_classes) * self.num_samples_per_cond
        )

        with temp_seed(3407):
            if self.num_classes > max_classes:
                self.classes = random.sample(range(self.num_classes), max_classes)
            else:
                self.classes = range(self.num_classes)
        self.samples = [
            self.classes[i // num_samples_per_cond] for i in range(self.num_samples)
        ]
        self.generator = generator
        self.x_N = torch.randn(
            self.num_samples, *shape, generator=self.generator, dtype=torch.float32
        )

    def __getitem__(self, index):
        class_idx = self.samples[index]
        embedding = (
            torch.nn.functional.one_hot(
                torch.ones(1, dtype=torch.int64) * class_idx, self.num_classes
            )
            .squeeze(0)
            .float()
        )

        return {"x_N": self.x_N[index], "name": str(class_idx), "label": embedding}

    def __len__(self):
        return self.num_samples


class TextCondPromptBed(Dataset):
    def __init__(
        self,
        path,
        num_samples_per_cond,
        text_embedding_name=None,
        generator=None,
        shape=(3, 32, 32),
    ):
        self.path = path
        self.num_samples_per_cond = num_samples_per_cond
        self.metadata = pd.read_csv(path / Path("metadata.csv"))
        self.num_samples = len(self.metadata) * num_samples_per_cond
        self.text_embedding_name = text_embedding_name
        self.generator = generator
        self.x_N = torch.randn(
            self.num_samples, *shape, generator=self.generator, dtype=torch.float32
        )

    def __getitem__(self, index):
        metadata = self.metadata.iloc[index // self.num_samples_per_cond]
        text = metadata["text"]
        if self.text_embedding_name is None:
            return {"x_N": self.x_N[index], "text": text}
        else:
            embedding = torch.from_numpy(
                np.load(
                    self.path
                    / Path(f"{self.text_embedding_name}_embeddings")
                    / Path(metadata["file_name"])
                )
            )
            return {
                "x_N": self.x_N[index],
                "text": text,
                self.text_embedding_name: embedding,
            }

    def __len__(self):
        return self.num_samples


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
