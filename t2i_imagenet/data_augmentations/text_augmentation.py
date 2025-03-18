import torch
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from typing import Union
from tqdm import tqdm
import argparse

from text_utils import (
    CaptioningDataset
)



class LlavaCaptioner:
    def __init__(self,
        dataset: CaptioningDataset,
        prompt: str = "Describe this image",
        max_new_tokens: int = 200,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert isinstance(
            dataset, CaptioningDataset
        ), f"dataset must be an 'CaptioningDataset' instance but got {type(self.dataset)}."
        self.dataset = dataset

        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self.model, self.processor = self.load_model()

    def load_model(self, model_path: str = "llava-hf/llama3-llava-next-8b-hf"):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(self.device)
        processor = LlavaNextProcessor.from_pretrained(
            model_path
        )
        return model, processor
    
    def generate(
        self, 
        images: Union[str, Image.Image, torch.Tensor],
    ) -> str:
        conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image"},
                    ],
                },
            ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=images, text=[prompt] * len(images), return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return captions
    
    def caption(self, res_file: str = "captions.tsv", save_on_the_go: bool = False):
        all_captions = []
        for i, (image, image_path) in enumerate(tqdm(self.dataset)):
            captions = self.generate(image)
            if save_on_the_go:
                with open(f"{res_file}", "a") as f:
                    for j in range(len(image_path)):
                        f.write(
                            f"{str(image_path[j]).split('/')[-1]}\t{self.postprocess_caption(captions[j])}\n"
                        )
            else:
                all_captions.append((image_path, captions))
        
        if not save_on_the_go:
            with open(f"{res_file}", "w") as f:
                for image_path, captions in all_captions:
                    for j in range(len(image_path)):
                        f.write(
                            f"{str(image_path[j]).split('/')[-1]}\t{self.postprocess_caption(captions[j])}\n"
                        )

    def postprocess_caption(self, caption: str):
        splitted = caption.split("assistant")
        caption = splitted[-1]
        caption = caption.replace("\n\n\n", "")
        caption = caption.replace("\n", "")
        if "." in caption:
            last_dot_index = caption.rfind(".")
            caption = caption[:last_dot_index]
        caption = caption + "."
        print(caption)
        return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", help="path to the folder containing the images to be captioned"
    )
    parser.add_argument(
        "--batch_size", help="batch size to be used for captioning", default=4
    )
    parser.add_argument(
        "--save_on_the_go",
        help="save the captions on the go or at the end of the captioning process",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--res_file",
        help="path to the file where the captions will be saved. Should be a .tsv file stored in the same folder as the image folder", 
        default="captions.tsv",
    )
    parser.add_argument(
        "--prompt",
        help="prompt to be used for captioning",
        default="Describe this image",
    )
    parser.add_argument(
        "--max_new_tokens",
        help="maximum number of tokens to be generated for the caption",
        default=200,
    )
    args = parser.parse_args()

    dataset = CaptioningDataset(
       args.image_dir, batch_size=int(args.batch_size),
    )
    captioner = LlavaCaptioner(
        dataset=dataset,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    caption = captioner.caption(res_file=args.res_file, save_on_the_go=False)