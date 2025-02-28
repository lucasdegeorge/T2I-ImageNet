import argparse

import numpy as np
import torch


def hotfix_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    embedding = torch.from_numpy(np.load("utils/flan_t5_xl_uncond.npy"))
    checkpoint["state_dict"]["uncond_conditioning"] = embedding
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint unconditional conditioning embedding hotfixed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    args = parser.parse_args()
    hotfix_checkpoint(args.checkpoint_path)
