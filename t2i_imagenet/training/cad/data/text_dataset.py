import json
import logging
import os
import random
from collections import OrderedDict
from functools import partial
from multiprocessing import Value
from pathlib import Path

import braceexpand
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from lightning_fabric.utilities.rank_zero import _get_rank
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)


class TextDataset(Dataset):
    def __init__(
        self,
        root,
        image_transforms=None,
        return_clip=False,
        clip_name="clip_score",
        text_embedding_name=None,
        vae_embedding_name_mean=None,
        vae_embedding_name_std=None,
        return_image=True,
        return_text=True,
    ):
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.metadata = pd.read_csv(
            root / Path("global_metadata.csv"), converters={"key": str}
        )
        self.clip_name = clip_name
        self.return_clip = return_clip
        self.text_embedding_name = text_embedding_name
        self.vae_embedding_name_mean = vae_embedding_name_mean
        self.vae_embedding_name_std = vae_embedding_name_std
        self.return_image = return_image
        self.return_text = return_text

    def __getitem__(self, idx):
        return_dict = {}
        metadata = self.metadata.iloc[idx]
        key = metadata["key"]
        if self.return_image:
            image_path = self.root / "images" / f"{key}.jpg"
            image = Image.open(image_path).convert("RGB")
            if self.image_transforms is not None:
                image = self.image_transforms(image)
            return_dict["image"] = image
        if self.return_text:
            caption = metadata["caption"]
            return_dict["text"] = caption
        if self.text_embedding_name is not None:
            text_embedding_path = (
                self.root / f"{self.text_embedding_name}_embeddings" / f"{key}.npy"
            )
            text_embedding = torch.from_numpy(np.load(text_embedding_path))
            return_dict[f"{self.text_embedding_name}"] = text_embedding
        if self.vae_embedding_name_mean is not None:
            vae_embedding_path = self.root / self.vae_embedding_name_mean / f"{key}.npy"
            vae_embedding = torch.from_numpy(np.load(vae_embedding_path))
            return_dict[self.vae_embedding_name_mean] = vae_embedding
            if self.vae_embedding_name_std is not None:
                vae_embedding_std_path = (
                    self.root / self.vae_embedding_name_std / f"{key}.npy"
                )
                vae_embedding_std = torch.from_numpy(np.load(vae_embedding_std_path))
                return_dict[self.vae_embedding_name_std] = vae_embedding_std
        if self.return_clip:
            clip_score = metadata[self.clip_name]
            return_dict["clip_score"] = clip_score
        return_dict["coherence"] = 1
        return return_dict

    def __len__(self):
        return len(self.metadata)


class TextWebDataset(wds.DataPipeline):
    def __init__(
        self,
        root,
        image_transforms=None,
        distributed=True,
        train=True,
        epoch=0,
        seed=3407,
        text_embedding_name=None,
        vae_embedding_name_mean=None,
        vae_embedding_name_std=None,
        return_image=True,
        return_text=True,
        return_coherence=False,
        min_image_size=256,
        bin_coherence=False,
        num_bins=8,
        clip_filter_threshold=0.0,
        aesthetic_threshold=0.0,
        shard_shuffle_size=256,
        shard_shuffle_initial=128,
        sample_shuffle_size=1000,
        sample_shuffle_initial=100,
    ):
        self.image_transforms = image_transforms
        dataset_tar_files = []
        self.clip_filter_threshold = clip_filter_threshold
        # Get a list of all tar files in the directory
        if " " in root:
            root = root.split(" ")
            print(f"Using multiple dataset[s: {root}")
        if isinstance(root, str):
            tar_files = [f for f in os.listdir(root) if f.endswith(".tar")]

            # Sort the list of tar files
            tar_files.sort()

            first_tar_file = tar_files[0].split(".")[0]
            last_tar_file = tar_files[-1].split(".")[0]

            for tar_file in tar_files:
                dataset_tar_files.append(f"{root}/{tar_file}")

            dataset_pattern = f"{root}/{{{first_tar_file}..{last_tar_file}}}.tar"
            self.num_samples, _ = get_dataset_size(
                dataset_pattern,
                min_image_size,
                clip_filter_threshold,
                aesthetic_threshold,
            )
        elif isinstance(root, list):
            num_samples = 0
            for r in root:
                tar_files = [f for f in os.listdir(r) if f.endswith(".tar")]
                tar_files.sort()
                first_tar_file = tar_files[0].split(".")[0]
                last_tar_file = tar_files[-1].split(".")[0]

                for tar_file in tar_files:
                    dataset_tar_files.append(f"{r}/{tar_file}")

                num_samples += get_dataset_size(
                    f"{r}/{{{first_tar_file}..{last_tar_file}}}.tar",
                    min_image_size,
                    clip_filter_threshold,
                    aesthetic_threshold,
                )[0]
            self.num_samples = num_samples
        else:
            raise ValueError(
                f"root must be a string or list of strings. Got {type(root)}"
            )
        rank = _get_rank()
        if return_coherence:
            if bin_coherence:
                if rank is None:
                    dataset_for_coherence = (
                        wds.WebDataset(dataset_tar_files).shuffle(1000).to_tuple("json")
                    )
                    coherences = []
                    MAX_SAMPLES_BIN_ESTIMATION = 100000
                    for i, (metadata,) in enumerate(tqdm(dataset_for_coherence)):
                        metadata = json.loads(metadata)
                        if i > MAX_SAMPLES_BIN_ESTIMATION:
                            break
                        coherences.append(metadata["clip_score"] / 100)
                    bins = torch.quantile(
                        torch.tensor(coherences),
                        q=torch.linspace(0, 1, num_bins + 1),
                    )
                    bins[0] = 0.0
                    bins[-1] = 1.0
                    self.bins = bins
                    print(f"Computed bins: {self.bins}")
                else:
                    dataset_for_coherence = wds.DataPipeline(
                        wds.SimpleShardList(dataset_tar_files),
                        detshuffle2(
                            bufsize=shard_shuffle_size,
                            initial=shard_shuffle_initial,
                            seed=seed,
                        ),
                        wds.split_by_node,
                        wds.split_by_worker,
                        tarfile_to_samples_nothrow,
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        ),
                        wds.to_tuple("json"),
                    )
                    coherences = []
                    MAX_SAMPLES_BIN_ESTIMATION = 100000
                    for i, (metadata,) in enumerate(tqdm(dataset_for_coherence)):
                        metadata = json.loads(metadata)
                        if i > MAX_SAMPLES_BIN_ESTIMATION:
                            break
                        coherences.append(metadata["clip_score"] / 100)
                    # coherences = torch.cat(
                    #     torch.distributed.nn.all_gather(torch.tensor(coherences)), dim=0
                    # )
                    bins = torch.quantile(
                        torch.tensor(coherences),
                        q=torch.linspace(0, 1, num_bins + 1),
                    )
                    bins[0] = 0.0
                    bins[-1] = 1.0
                    self.bins = bins
                    print(f"Computed bins: {self.bins}")

        self.shared_epoch = SharedEpoch(epoch)
        pipeline = [wds.SimpleShardList(dataset_tar_files)]

        if distributed:
            pipeline.extend(
                [
                    (
                        detshuffle2(
                            bufsize=shard_shuffle_size,
                            initial=shard_shuffle_initial,
                            seed=seed,
                            epoch=self.shared_epoch,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    (
                        wds.shuffle(
                            bufsize=shard_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )

        pipeline.extend(
            [
                wds.select(filter_no_caption_or_no_image),
                wds.select(
                    partial(
                        filter_metadata,
                        min_image_size=min_image_size,
                        min_clip_score=clip_filter_threshold,
                        min_aesthetic=aesthetic_threshold,
                    )
                ),
            ]
        )
        outputs_transforms = OrderedDict()
        outputs_rename = OrderedDict()
        if return_image:
            outputs_rename["image.jpg"] = "jpg;png;webp;jpeg"
            outputs_transforms["image.jpg"] = self.image_transforms
        if vae_embedding_name_mean is not None:
            outputs_rename[
                f"{vae_embedding_name_mean}.npy"
            ] = f"{vae_embedding_name_mean}.npy"
            outputs_transforms[
                f"{vae_embedding_name_mean}.npy"
            ] = lambda x: torch.from_numpy(x)
        if vae_embedding_name_std is not None:
            outputs_rename[
                f"{vae_embedding_name_std}.npy"
            ] = f"{vae_embedding_name_std}.npy"
            outputs_transforms[
                f"{vae_embedding_name_std}.npy"
            ] = lambda x: torch.from_numpy(x)
        if return_text:
            outputs_rename["text.txt"] = "txt"
            outputs_transforms["text.txt"] = lambda x: x
        if text_embedding_name is not None:
            outputs_rename[
                f"{text_embedding_name}.npy"
            ] = f"{text_embedding_name}_embeddings.npy"
            outputs_transforms[
                f"{text_embedding_name}.npy"
            ] = lambda x: torch.from_numpy(x)
        if return_coherence:
            outputs_rename["coherence.json"] = "json"
            outputs_transforms["coherence.json"] = (
                (
                    lambda x: (
                        torch.bucketize(
                            torch.tensor(x["clip_score"] / 100, dtype=torch.float32),
                            self.bins,
                        ).item()
                        - 1
                    )
                    / (len(self.bins) - 2)
                )
                if bin_coherence
                else lambda x: torch.tensor(x["clip_score"], dtype=torch.float32) / 100
            )
        pipeline.extend(
            [
                wds.rename(**outputs_rename),
                filter_dict_keys(*outputs_rename.keys(), handler=log_and_continue),
            ]
        )
        if return_image:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        else:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        pipeline.extend(
            [
                wds.map_dict(**outputs_transforms, handler=log_and_continue),
                wds.rename(
                    **{k.split(".")[0]: k for k in outputs_transforms.keys()},
                ),
            ]
        )

        super().__init__(*pipeline)

    def get_clip_score_from_meta(self, metadata):
        return metadata[self.clip_name]

    def __len__(self):
        return self.num_samples


# Modified from open_clip


def get_dataset_size(
    shards, min_image_size, clip_filter_threshold, aesthetic_threshold
):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(
        dir_path,
        f"sizes_{min_image_size}_{str(clip_filter_threshold).replace('.', '_')}_{str(aesthetic_threshold).replace('.', '_')}.json",
    )
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    else:
        total_size = 0  # num samples undefined
        sizes = {}
        for shard in tqdm(shards_list):
            dataset = (
                wds.WebDataset(shard)
                .select(
                    partial(
                        filter_metadata,
                        min_image_size=min_image_size,
                        min_clip_score=clip_filter_threshold,
                        min_aesthetic=aesthetic_threshold,
                    )
                )
                .to_tuple("__key__")
            )
            num_samples = sum(1 for _ in dataset)
            total_size += num_samples
            sizes[os.path.basename(shard)] = num_samples
        print(f"Total number of samples: {total_size}")
        with open(sizes_filename, "w") as f:
            json.dump(sizes, f)

    num_shards = len(shards_list)
    return total_size, num_shards


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


# _SHARD_SHUFFLE_SIZE = 256
# _SHARD_SHUFFLE_INITIAL = 128
# _SAMPLE_SHUFFLE_SIZE = 5000
# _SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        try:
            fname, value = filesample["fname"], filesample["data"]
        except KeyError as exn:
            print(f"Error with filesample {filesample} - exception: {exn}")
            continue
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def filter_metadata(sample, min_image_size, min_clip_score, min_aesthetic):
    if "json" in sample.keys():
        metadata = json.loads(sample["json"])
        width = metadata["width"]
        height = metadata["height"]
        clip_score = metadata["clip_score"] / 100
        aesthetic = metadata["aesthetic_score"]
        return (
            width >= min_image_size
            and height >= min_image_size
            and clip_score >= min_clip_score
            and aesthetic >= min_aesthetic
        )
    else:
        return True


def _filter_dict_keys(
    data,
    *args,
    handler=wds.reraise_exception,
    missing_is_error=True,
    none_is_error=None,
):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = {
                f: wds.getfirst(sample, f, missing_is_error=missing_is_error)
                for f in args
            }
            print
            if none_is_error and any(x is None for x in result):
                raise ValueError(f"to_tuple {args} got {sample.keys()}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


filter_dict_keys = wds.pipelinefilter(_filter_dict_keys)
