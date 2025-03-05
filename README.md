<div align="center">

# How far can we go with ImageNet for Text-to-Image generation?

<a href="https://lucasdegeorge.github.io/" >Lucas Degeorge</a>, <a href="https://arijit-hub.github.io/" >Arijit Ghosh</a>, <a href="https://nicolas-dufour.github.io/" >Nicolas Dufour</a>, <a href="https://davidpicard.github.io/" >David Picard</a>, <a href="https://vicky.kalogeiton.info/" >Vicky Kalogeiton</a>
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


## Pretrained models

To use the pre-trained models, do the following:
```python
from pipe import T2IPipeline
pipe = T2IPipeline("Lucasdegeorge/CAD-I").to("cuda")
prompt = "An adorable otter, with its sleek, brown fur and bright, curious eyes, playfully interacts with a vibrant bunch of broccoli... "
image = pipe(prompt, cfg=15)
```

Models are hosted in the HuggingFace Hub. Weights can be found at:

- **CAD-I**: model trained with both **Text and Image Augmentation**: https://huggingface.co/Lucasdegeorge/CAD-I

- **CAD-I_TA**: model trained with **Text Augmentation only**: https://huggingface.co/Lucasdegeorge/CAD-I_TA

More details about the `T2IPipeline` are given in the [model card on HuggingFace](https://huggingface.co/Lucasdegeorge/CAD-I)

### Prompts

Our models have been specifically trained to handle very long and detailed prompts. To get the best performance and results, we encourage you to use them with prompts that are rich in detail. Short or vague prompts may not fully utilize the model's capabilities.
Example prompts: 
```
A majestic elephant stands tall and proud in the heart of the African savannah, its wrinkled, gray skin glistening under the intense afternoon sun. The elephant's large, flapping ears and long, sweeping trunk create a sense of grace and power as it gently sways, surveying the vast, golden grasslands stretching out before it. In the distance, a herd of zebras grazes peacefully, their stripes blending with the tall, dry grass. The scene is completed by a lone acacia tree silhouetted against the setting sun, casting long, dramatic shadows across the landscape.
A classic film camera rests on a tripod, its worn leather strap and scratched metal body telling the story of countless adventures and captured moments. The camera is positioned in a scenic landscape, with rolling hills, a winding river, and a distant mountain range bathed in the soft, golden light of sunset. In the foreground, a wildflower meadow sways gently in the breeze, while the camera's lens captures the beauty and tranquility of the scene, preserving it for eternity.
A graceful flamingo stands elegantly in the shallow waters of a tranquil lagoon, its vibrant pink feathers reflecting beautifully in the still water. The flamingo's long, slender legs and curved neck create a picture of poise and balance as it dips its beak into the water, searching for food. Behind the flamingo, a lush mangrove forest stretches out, its dense foliage providing a rich habitat for various wildlife. The scene is completed by a clear blue sky and the gentle rustling of leaves in the breeze
A hearty, overstuffed sandwich sits on a wooden cutting board, its layers of fresh, crisp lettuce, juicy tomatoes, and thinly sliced deli meats peeking out from between two slices of golden-brown bread. The sandwich's tantalizing aroma fills the air, mingling with the scent of freshly baked bread and tangy mustard. In the background, a bustling deli comes to life, with shelves lined with jars of pickles, a gleaming meat slicer, and a chalkboard menu listing the day's specials. The scene is completed by the lively chatter of customers and the clinking of glasses.
A stunning oil painting of a majestic tiger hangs on the wall of a dimly-lit art gallery, its vibrant colors and intricate details drawing the viewer in. The tiger's powerful, muscular body is depicted in mid-stride, its stripes blending seamlessly with the lush jungle foliage surrounding it. The painting captures the tiger's intense, amber eyes and the subtle play of light and shadow on its fur, creating a sense of depth and movement. The background features a dense canopy of trees and a cascading waterfall, adding to the wild, untamed atmosphere of the scene.
A clever magpie perched on a rustic wooden fence post, its iridescent black and white feathers shimmering in the sunlight. The bird tilts its head, holding a shiny trinket in its beak, with a backdrop of a golden wheat field swaying gently in the breeze. A few more curios and found objects are scattered along the fence, hinting at the magpie's treasure trove hidden nearby. A clear blue sky with puffy white clouds completes the scenic countryside atmosphere.
A playful dolphin leaps gracefully out of the sparkling turquoise waters, its sleek, gray body arcing through the air before diving back into the waves with a splash. Nearby, a classic wooden sailboat glides smoothly across the ocean, its white sails billowing in the breeze. The dolphin swims alongside the boat, its joyful antics mirrored by the shimmering sunlight dancing on the water's surface. The scene is completed by a clear blue sky and the distant horizon, where the sea meets the sky.
```

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

