import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="t2i_imagenet",
    version="1.0",
    description="Package for the paper How far can we go with ImageNet for Text-to-Image generation?",
    author="Lucas Degeorge and Arijit Ghosh",
    # packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch==2.4.0",
        "torchvision==0.19.0",
        "einops",
        "transformers",
        "diffusers",
        "huggingface_hub",
        "pandas",
        "sentencepiece",
        "tqdm",
        "webdataset==0.2.57"
    ],
    packages=["t2i_imagenet"],
    include_package_data=True,
    extras_require={
        "train": [
            "hydra-core",
            "lightning",
            "torch-fidelity",
            "pandas",
            "wandb",
            "timm",
            "scikit-learn",
            "webdataset==0.2.57",
            "lovely_tensors",
        ],
    },
)
