import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def main(
    src, 
    dest,
    img_size, 
    vae_embeddings_mean_name="vae_embeddings_mean",
    vae_embeddings_std_name="vae_embeddings_std",
):
    all_shard = [str(tar) for tar in list(Path(src).glob("*.tar"))]
    keys_wd = [
        "__key__",
        "jpg",
        "txt",
        # "label.json",
        "flan_t5_xl_embeddings.npy",
        f"vae_embeddings_mean_{img_size}.npy",
        f"vae_embeddings_std_{img_size}.npy"
    ]

    webdataset = wds.WebDataset(all_shard).to_tuple(*keys_wd)

    dest_dir = Path(dest)
    dest_dir.mkdir(exist_ok=True, parents=True)
    images_dir = dest_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    flan_t5_xl_embeddings_dir = dest_dir / "flan_t5_xl_embeddings"
    flan_t5_xl_embeddings_dir.mkdir(exist_ok=True, parents=True)

    vae_embeddings_mean_dir = dest_dir / f"{vae_embeddings_mean_name}_{img_size}" 
    vae_embeddings_mean_dir.mkdir(exist_ok=True, parents=True)
    vae_embeddings_std_dir = dest_dir / f"{vae_embeddings_std_name}_{img_size}"
    vae_embeddings_std_dir.mkdir(exist_ok=True, parents=True)

    global_metadata = []
    for i, sample in enumerate(tqdm(webdataset)):
        key, image, text, flan_t5_xl_embeddings, vae_embeddings_mean, vae_embeddings_std = sample
        
        with open(images_dir / f"{key}.jpg", "wb") as f:
            f.write(image)
        with open(flan_t5_xl_embeddings_dir / f"{key}.npy", "wb") as f:
            f.write(flan_t5_xl_embeddings)
        with open(vae_embeddings_mean_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_mean)
        with open(vae_embeddings_std_dir / f"{key}.npy", "wb") as f:
            f.write(vae_embeddings_std)

        # Append to dataframe
        global_metadata.append({"key": str(key), "caption": text.decode('utf-8'),})

    global_metadata = pd.DataFrame(global_metadata)
    global_metadata["key"] = global_metadata["key"].astype(str)
    global_metadata = global_metadata.reset_index(drop=True)
    global_metadata.to_csv(dest_dir / "global_metadata.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="path to source files")
    parser.add_argument("--dest", help="path to destination files")
    parser.add_argument("--img_size", default=256, type=int)
    args = parser.parse_args()

    vae_embeddings_mean_name = "vae_embeddings_mean"
    vae_embeddings_std_name = "vae_embeddings_std"

    main(
        args.src, 
        args.dest, 
        int(args.img_size),
        vae_embeddings_mean_name=vae_embeddings_mean_name,
        vae_embeddings_std_name=vae_embeddings_std_name
    )
