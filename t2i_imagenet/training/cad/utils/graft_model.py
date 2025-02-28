import os
import sys

# Ajouter le répertoire racine au chemin
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

import math
from copy import deepcopy
from pathlib import Path

import einops
import hydra
import torch

from models.diffusion import DiffusionModule


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    model = DiffusionModule(cfg.model)
    print(f"Loading model from {cfg.grafting_state_dict_path}")
    grafting_state_dict = torch.load(cfg.grafting_state_dict_path, map_location="cpu")

    num_initial_blocks = len(
        set(
            [
                key.split(".")[2]
                for key in grafting_state_dict["state_dict"].keys()
                if "network.rin_blocks" in key
            ]
        )
    )
    target_state_dict = deepcopy(grafting_state_dict)
    if num_initial_blocks + 2 == cfg.model.network.num_blocks:
        print("Grafting 2 blocks")
        for key in list(grafting_state_dict["state_dict"].keys()):
            if "network.rin_blocks" in key:
                rin_block_number = int(key.split(".")[2])
                splited_key = key.split(".")
                if rin_block_number == num_initial_blocks - 1:
                    splited_key[2] = str(rin_block_number + 1)
                    new_key = ".".join(splited_key)
                    target_state_dict["state_dict"][new_key] = grafting_state_dict[
                        "state_dict"
                    ][key]
                    splited_key[2] = str(rin_block_number + 2)
                    new_key = ".".join(splited_key)
                    target_state_dict["state_dict"][new_key] = grafting_state_dict[
                        "state_dict"
                    ][key]
                else:
                    splited_key[2] = str(rin_block_number + 1)
                    new_key = ".".join(splited_key)
                    target_state_dict["state_dict"][new_key] = grafting_state_dict[
                        "state_dict"
                    ][key]
        grafting_state_dict = target_state_dict

    elif num_initial_blocks == cfg.model.network.num_blocks:
        pass
    else:
        raise ValueError(
            f"Number of blocks in the model should be same as initial or initial + 2"
        )
    num_grafted_latents = grafting_state_dict["state_dict"]["network.latents"].shape[0]
    if model.network.latents.shape[0] > num_grafted_latents:
        print("Grafting latents")
        new_latents = model.network.latents.detach()
        new_latents[:num_grafted_latents] = grafting_state_dict["state_dict"][
            "network.latents"
        ]
        grafting_state_dict["state_dict"]["network.latents"] = new_latents

        new_latents_ema = model.network.latents.detach()
        new_latents_ema[:num_grafted_latents] = grafting_state_dict["state_dict"][
            "ema_network.latents"
        ]
        grafting_state_dict["state_dict"]["ema_network.latents"] = new_latents_ema

    pos_emb = grafting_state_dict["state_dict"]["network.data_pos_embedding"]
    pos_emb_ema = grafting_state_dict["state_dict"]["ema_network.data_pos_embedding"]
    height = int(math.sqrt(pos_emb.shape[0]))
    pos_emb = einops.rearrange(pos_emb, "(h w) c -> c h w", h=height, w=height)
    pos_emb_ema = einops.rearrange(pos_emb_ema, "(h w) c -> c h w", h=height, w=height)
    pos_emb = torch.nn.functional.interpolate(
        pos_emb.unsqueeze(0),
        scale_factor=2,
        mode="bilinear",
    )
    pos_emb_ema = torch.nn.functional.interpolate(
        pos_emb_ema.unsqueeze(0),
        scale_factor=2,
        mode="bilinear",
    )
    pos_emb = einops.rearrange(pos_emb.squeeze(0), "c h w -> (h w) c")
    pos_emb_ema = einops.rearrange(pos_emb_ema.squeeze(0), "c h w -> (h w) c")
    grafting_state_dict["state_dict"]["network.data_pos_embedding"] = pos_emb
    grafting_state_dict["state_dict"]["ema_network.data_pos_embedding"] = pos_emb_ema
    model.load_state_dict(grafting_state_dict["state_dict"], strict=False)
    Path(cfg.checkpoints.dirpath).mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        Path(cfg.checkpoints.dirpath) / "init.ckpt",
    )


if __name__ == "__main__":
    main()
