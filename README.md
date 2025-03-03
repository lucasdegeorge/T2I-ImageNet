<div align="center">

# How far can we go with ImageNet for Text-to-Image generation?

<a href="https://lucasdegeorge.github.io/" >Lucas Degeorge</a>, <a href="https://arijit-hub.github.io/" >Arijit Ghosh</a>, <a href="https://nicolas-dufour.github.io/" >Nicolas Dufour</a>, <a href="https://vicky.kalogeiton.info/" >Vicky Kalogeiton</a>, <a href="https://davidpicard.github.io/" >David Picard</a>
</div>

![DED](/assets/teaser.png)

This repo has the code for the paper "How far can we go with ImageNet for Text-to-Image generation?"

The core idea is that text-to-image generation models typically rely on vast datasets, prioritizing quantity over quality. The usual solution is to gather massive amounts of data. We propose a new approach that leverages strategic data augmentation of small, well-curated datasets to enhance the performance of these models. We show that this method improves the quality of the generated images on several benchmarks.

Paper on Arxiv: https://arxiv.org/pdf/2502.21318

Project website: https://lucasdegeorge.github.io/projects/t2i_imagenet/


## Install 

To install, first create a virtual environment with python (at least 3.9) and run 

```bash
pip install -e .
```

If you want to use the training pipeline (see [`training/README.md`](https://github.com/lucasdegeorge/T2I-ImageNet/blob/main/t2i_imagenet/training/README.md)):

```bash
pip install .[train]
```

Depending on your CUDA version, be careful installing torch.


## Text and Pixel Augmentation recipe

See [`data_augmentations/README.md`](https://github.com/lucasdegeorge/T2I-ImageNet/blob/main/t2i_imagenet/data_augmentations/README.md)

## Training

See [`training/README.md`](https://github.com/lucasdegeorge/T2I-ImageNet/blob/main/t2i_imagenet/training/README.md)

## Citation
If you happen to use this repo in your experiments, you can acknowledge us by citing the following paper:

```bibtex
@article{degeorge2025farimagenettexttoimagegeneration, 
     title           ={How far can we go with ImageNet for Text-to-Image generation?}, 
     author          ={Lucas Degeorges and Arijit Ghosh and Nicolas Dufour and David Picard and Vicky Kalogeiton}, 
     year            ={2025}, 
     journal         ={arXiv},
 }
```

