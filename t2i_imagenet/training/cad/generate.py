import torch
import argparse
from PIL import Image
import numpy as np
import random
import os
from pathlib import Path
from hydra.core.global_hydra import GlobalHydra
import hydra
from einops import rearrange
import json

from models.diffusion import DiffusionModule

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(ckpt_name: str, config: dict, ckpt_path: str=None) -> DiffusionModule:
    GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=f"../cad/configs")
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "root_dir=./",
            f"experiment_name={ckpt_name}",
            "model/precomputed_text_embeddings='no'",
            *[f"{k}={v}" for k, v in config.items()],

        ],
    )
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
    else:
        ckpt_path = Path(cfg.checkpoints.dirpath) / Path("last.ckpt")
    model = DiffusionModule.load_from_checkpoint(
        ckpt_path,
        strict=False,
        cfg=cfg.model,
    ).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    for param in model.parameters():
        param.requires_grad = False

    return model


def generate(
    model: DiffusionModule,
    prompts: list,
    seed: int = 0,
    guidance_scale: float = 10.0,
    num_inference_steps: int = 250,
    num_images_per_prompt: int = 1,
    clamp_value: float = 1.0,
    sampler="val",
) -> Image.Image:
    seed_everything(seed)
    if num_images_per_prompt > 1:
        prompts = [
            prompt for prompt in prompts for _ in range(num_images_per_prompt)
        ]
    images = model.sample(
        batch_size= len(prompts),
        shape=(4, 32, 32),
        cond=prompts,
        cfg=guidance_scale,
        num_steps=num_inference_steps,
        stage=sampler,
        clamp_value=clamp_value,
    )
    if num_images_per_prompt > 1:
        images = rearrange(
            images, "(b1 b2) c h w -> b1 c h (b2 w)", b2=num_images_per_prompt
        )
    return images

def main(args):
    # Prompts
    assert args.prompt_file or args.prompt, "Either prompt_file or prompt must be provided"
    assert not (args.prompt_file and args.prompt), "Only one of prompt_file or prompt must be provided"
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file, "r") as f:
            prompts = f.read().splitlines()

    # Model loading
    config = json.loads(args.config) if args.config else {}
    config["overrides"] = args.overrides
    model = load_model(args.model_ckpt, config, args.ckpt_path)

    # Save directory
    save_folder_path = Path(
            f"{args.save_folder_path}/{args.model_ckpt}"
        )
    os.makedirs(save_folder_path, exist_ok=True)

    # Generating images
    with torch.no_grad():
        images = generate(
            model,
            prompts,
            seed=3407,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.num_sampling_steps,
            num_images_per_prompt=args.num_samples_per_prompt,
            clamp_value=args.clamp_value,
            sampler=args.sampler,
        )
    images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    images = [Image.fromarray(image) for image in images]
    for i, image in enumerate(images):
        image.save(
            f"{save_folder_path}/{int(i):06d}.png"
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, help="Model checkpoint name")
    parser.add_argument("--overrides", type=str, help="Overrides for the config")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Hydra config with parameters to override. Should be a JSON configuration string"
    )
    parser.add_argument("--prompt_file", type=str, help="File (.txt) file containing prompts")
    parser.add_argument("--prompt", type=str, help="Prompt")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint directory. If not provided, the checkpoint will be loaded from 'cad/checkpoints/model_name/last.ckpt'",
    )
    parser.add_argument(
        "--save_folder_path",
        type=str,
        default="cad/images",
    )
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--sampler", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=10.0)
    parser.add_argument("--clamp_value", type=float, default=1.0)

    args = parser.parse_args()
    main(args)