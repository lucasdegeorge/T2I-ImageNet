import webdataset as wds
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import AutoencoderKL
import csv
import json


def create_tar_files(
    src, 
    dest,
    compute_t5_embeddings=False,
    compute_vae_embeddings=False,
    caption_name_file='captions', 
    images_folder_name='images', 
    samples_per_tar=1000
):
    caption_file = os.path.join(src, f'{caption_name_file}.tsv')
    if not os.path.exists(caption_file):
        caption_file = os.path.join(src, f'{caption_name_file}.json')
    captions = read_captions(caption_file)
    images_folder = os.path.join(src, images_folder_name)
    image_files = get_image_paths(images_folder)
    print(f'Found {len(image_files)} images')

    for i in range(0, len(image_files), samples_per_tar):
        tar_file = os.path.join(dest, f'{i//samples_per_tar:04d}.tar')
        with wds.TarWriter(tar_file) as tar:
            for image_path in tqdm(image_files[i:i+samples_per_tar]):
                file_name = os.path.basename(image_path)
                try:
                    caption = captions[file_name]
                except KeyError:
                    try:
                        caption = captions[file_name.split('.')[0]]
                    except KeyError:
                        print(f'No caption found for {file_name}')
                        continue

                jpg_data = open(image_path, 'rb').read()

                sample = {
                    '__key__': file_name.split('.')[0],
                    'jpg': jpg_data,
                    'txt': caption
                }

                if compute_t5_embeddings:
                    assert t5_model is not None and t5_tokenizer is not None, "T5 model and tokenizer must be loaded"
                    with torch.no_grad():
                        input_ids = t5_tokenizer(
                            caption,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=256,
                        ).to(t5_model.device)
                        t5_embeddings = (
                            t5_model(**input_ids).last_hidden_state.squeeze(0).detach().cpu().numpy()
                        )
                    sample['flan_t5_xl_embeddings.npy'] = t5_embeddings

                if compute_vae_embeddings:
                    assert vae_model is not None, "VAE model must be loaded"
                    with torch.no_grad():
                        image = Image.open(image_path).convert("RGB")
                        # image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().to(vae_model.device)
                        image = image_transforms(image).unsqueeze(0).to(vae_model.device)
                        mean_embeddings = vae_model.encode(image).latent_dist.mean.cpu().numpy()
                        std_embeddings = vae_model.encode(image).latent_dist.std.cpu().numpy()
                    sample[f'vae_embeddings_mean_{size_vae_embeddings}.npy'] = mean_embeddings.squeeze()
                    sample[f'vae_embeddings_std_{size_vae_embeddings}.npy'] = std_embeddings.squeeze()
                tar.write(sample)
        print(f'Tar file created at {tar_file}')


def read_captions(caption_file):
    print(f'Reading captions from {caption_file}')
    captions = {}
    if caption_file.endswith('.csv'):   
        with open(caption_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key, caption = row[0], row[1]
                    captions[key] = caption
        return captions
    elif caption_file.endswith('.tsv'):
        with open(caption_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    key, caption = row[0], row[1]
                    captions[key] = caption
        return captions
    elif caption_file.endswith('.json'):
        with open(caption_file, 'r') as f:
            captions = json.load(f)
        return captions


def load_t5_model(model_name='google/flan-t5-xl'):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer
    

def load_vae_model(model_path="stabilityai/sd-vae-ft-ema"):
    model = AutoencoderKL.from_pretrained(model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def get_image_paths(src, extension='.JPEG'):
    image_files = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(extension):
                image_files.append(os.path.join(root, file))
    return image_files


class CenterCrop(torch.nn.Module):
    def __init__(self, size=None, ratio="1:1"):
        super().__init__()
        self.size = size
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.size is None:
            if isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                w, h = img.size
            ratio = self.ratio.split(":")
            ratio = float(ratio[0]) / float(ratio[1])
            ratioed_w = int(h * ratio)
            ratioed_h = int(w / ratio)
            if w >= h:
                if ratioed_h <= h:
                    size = (ratioed_h, w)
                else:
                    size = (h, ratioed_w)
            else:
                if ratioed_w <= w:
                    size = (h, ratioed_w)
                else:
                    size = (ratioed_h, w)
        else:
            size = self.size
        return torchvision.transforms.functional.center_crop(img, size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files. Must contains a folder with images and a 'caption_file_name.csv' file with the captions")
    parser.add_argument("--dest", help="path to destination files")
    parser.add_argument("--compute_t5_embeddings", action='store_true', default=False)
    parser.add_argument("--compute_vae_embeddings", action='store_true', default=False)
    parser.add_argument("--size_vae_embeddings", help="size of the vae embeddings", default=256)
    parser.add_argument("--caption_file_name", help="name of the caption file", default='captions')
    parser.add_argument("--images_folder_name", help="name of the images folder", default='images')
    parser.add_argument("--samples_per_tar", help="number of samples per tar", default=1000)
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True, parents=True)
    samples_per_tar = int(args.samples_per_tar)
    if args.compute_t5_embeddings:
        t5_model, t5_tokenizer = load_t5_model()
    if args.compute_vae_embeddings:
        size_vae_embeddings = int(args.size_vae_embeddings)
        vae_model = load_vae_model()
        image_transforms = transforms.Compose(
            [
                CenterCrop(ratio="1:1"),
                transforms.Resize(size_vae_embeddings),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )

    create_tar_files(
        src,
        dest,
        args.compute_t5_embeddings,
        args.compute_vae_embeddings,
        args.caption_file_name,
        args.images_folder_name,
        samples_per_tar
    )