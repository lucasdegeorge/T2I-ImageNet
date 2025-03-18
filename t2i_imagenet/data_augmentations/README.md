## Text and Pixel augmentation based dataset creation

### Pixel augmented image creation

To generate the image augmented datasets, please ensure that you have your dataset stored in the following way:

```bash
|__root
    |__class_A
        |__img_A1.JPEG
        |__img_A2.JPEG
        |__...
    |__class_B
        |__img_B1.JPEG
        |__img_B2.JPEG
        |__...
    |_...
```

The image extensions allowed are `.JPEG`, `.jpg`, `.png`.

To create pixel augmented images using your own data use the following command:

```bash
python pixel_augmentation.py --base_image_size=256 \
    --setting=5 \
    --input_dir="/path/to/input/root/images" \
    --dataset_times=1 \
    --out_dir="/path/to/out/dir" \
    --img_extension="JPEG"
```

This will create `256x256` pixel augmented images based on CM<sup>all</sup> setting of same size as your original dataset and save it to `/path/to/out/dir`. Do change the arguments in accordance to your dataset and your preferred settings.


### Text augmentation 

To caption images using the LLaVA model, please ensure that all your images have one of the following extensions: `.JPEG`, `.jpg`, `.png` and are contained in one folder (that may contains several subfolders). 

Run the following command to caption your images. It will create `.tsv` file containing images paths and their captions.


```bash
python text_augmentation.py --image_dir="/path/to/input/root/images" \
    --batch_size=64 \
    --res_file="/path/to/input/root/images/captions.tsv" \
```


### Webdataset for training

To create a [Webdataset](https://github.com/webdataset/webdataset) with a text and image augmented dataset, please ensure that you have your dataset stored in the following way:


```bash
|__root
    |images
        |...
    |captions.tsv
```

Run the following command to create your Webdataset. It will create `.tar` files containing precomputed [latent](https://huggingface.co/stabilityai/sd-vae-ft-ema) embeddings of the images, precomputed [Flan-T5 XXL](https://huggingface.co/google/flan-t5-xxl) embeddings of the captions along with the raw images and the text captions. 


```bash
python create_wds.py --src="/path/to/input/root" \
    --dest="/path/to/output/root" \
    --compute_t5_embeddings \ 
    --compute_vae_embeddings \ 
    --size_vae_embeddings=256 
```


To convert a webdataset as a Pytorch Dataset, use the following command: 

```bash
python webdataset_to_dataset.py --src="/path/to/input/root" \
    --dest="/path/to/output/root" \
    --img_size=256
```
