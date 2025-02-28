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
            |__flan_t5_xl_embeddings
            |__vae_embeddings_mean
            |__vae_embeddings_std
    |_...
```

Use the files `data_augmentations/create_wds.py` and `data_augmentations/webdataset_to_dataset.py` to generate the webdatasets and the val folder, explained in `/data_augmentations/README.md`. Create a new config file in `configs/data`. 


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
    --prompt="An image of a dog playing chess"
```

To generate images with several prompts at once, replace the argument `--prompt` by `--prompt_file` and give the path to a `.txt` containing the prompts (separated by `\n`)

To override parameters in the Hydra config, use the argument `--config` with a JSON configuration string. For instance: 

```bash
python cad/generate.py --model_ckpt='model_name' \
    --overrides=imagenet_256_rin_base_text_ldm \
    --prompt="An image of a dog playing chess" \ 
    --config='{"model.channel_wise_normalisation": true, "model.data_preprocessing.vae_sample": true,  "model/precomputed_text_embeddings": "'no'"}' 
```

You can change generation parameters (cfg_scale, clamp_value, num_samples_per_prompt and num_sampling_steps) with the associated arguments. 