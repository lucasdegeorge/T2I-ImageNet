## Training diffusion models with CAD-I architecture

### Data preprocessing

First download and augment the dataset using our pixel and text augmentation, as explained in `/data_augmentations/README.md`. 

Store the datasets in /datasets in this way 

```bash
|__datasets
    |__custom_dataset
        |__train_wds
            |__00000.tar
            |__00001.tar
            |__...
        |__cutmix_wds
            |__00000.tar
            |__00001.tar
            |__...
        |__val
            |__images
                |_...
            |__flan_t5_xl_embeddings
                |_...
            |__vae_embeddings_mean
                |_...
            |__vae_embeddings_std
                |_...
            global_metadata.csv
    |_...
```

Use the `data_augmentations/create_wds.py` file to generate the webdatasets used for training (both for the raw and the cutmuix datasets). 

For the val set, first generate a webdataset using the `data_augmentations/create_wds.py` file. Then, use the `data_augmentations/webdataset_to_dataset.py` file to generate the val folder (see above) and the global_metadata.csv file. 

Please refer to the [`/data_augmentations/README.md`](https://github.com/lucasdegeorge/T2I-ImageNet/blob/main/t2i_imagenet/data_augmentations/README.md) for a detailed guidance of the `data_augmentations/create_wds.py` and `data_augmentations/webdataset_to_dataset.py` files. 

Finally, create a new dataset config file in `configs/data/dataset` where you specify the root arguments of the `train_instance`, `cutmix_instance` and `val_instance` (see `configs/data/dataset/imagenet_256_ldm_text_cutmix.yaml` as example). 


### Training


This training code is based around Hydra and requires to specify an override such as:

```bash
python cad/train.py overrides=config_file **Other hydra args**
```

To train a model with the basic config and both text and image augmentations, run the following command: 

```bash
python cad/train.py --overrides=imagenet_256_rin_base_text_ldm_cutmix
```

To train a model with the basic config and text-only augmentations, run the following command: 

```bash
python cad/train.py --overrides=imagenet_256_rin_base_text_ldm
```


### Generate images

To generate images with your trained model, run the following command:

```bash
python cad/generate.py --model_ckpt='model_name' \
    --overrides=imagenet_256_rin_base_text_ldm \
    --prompt="A majestic elephant stands tall and proud in the heart of the African savannah"
```

To generate images with several prompts at once, replace the argument `--prompt` by `--prompt_file` and give the path to a `.txt` containing the prompts (separated by `\n`)

To override parameters in the Hydra config, use the argument `--config` with a JSON configuration string. For instance: 

```bash
python cad/generate.py --model_ckpt='model_name' \
    --overrides=imagenet_256_rin_base_text_ldm \
    --prompt="A majestic elephant stands tall and proud in the heart of the African savannah" \ 
    --config='{"model.channel_wise_normalisation": true, "model.data_preprocessing.vae_sample": true,  "model/precomputed_text_embeddings": "'no'"}' 
```

You can change generation parameters (cfg_scale, clamp_value, num_samples_per_prompt and num_sampling_steps) with the associated arguments.


### Testing 

This repo allows to compute image quality metrics (FID, Precision, etc. see Table 1 of the paper). We refer you to the official repository of [GenEval](https://github.com/djghosh13/geneval) and [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) benchmarks for those metrics. (In particular check out this [installation note](https://github.com/djghosh13/geneval/issues/12)).


To test a model, use the following command: 

```bash
python cad/test.py --overrides=imagenet_256_rin_base_text_ldm_cutmix
```

Make sure you complete all the steps related to the evaluation dataset in [Data preprocessing](https://github.com/lucasdegeorge/T2I-ImageNet/blob/main/t2i_imagenet/training/README.md#data-preprocessing).