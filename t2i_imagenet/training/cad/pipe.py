import torch
import random
import string
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, T5EncoderModel
from huggingface_hub import PyTorchModelHubMixin
from models.networks.rin import CADRINTextCond
from models.samplers import (
    ddim_sampler,
    ddpm_sampler,
    dpm_sampler,
    dpm_plusplus_2S_sampler,
    dpm_plusplus_2M_sampler,
)

from models.postprocessing import (
    SD1_5VAEDecoderPostProcessing,
)

from models.schedulers import (
    SigmoidScheduler,
    LinearScheduler,
    CosineScheduler,
)
from models.preprocessing import TextPreprocessing
from models.preconditioning import DDPMPrecond

SAMPLERS = {
    "ddim": ddim_sampler,
    "ddpm": ddpm_sampler,
    "dpm": dpm_sampler,
    "dpm_2S": dpm_plusplus_2S_sampler,
    "dpm_2M": dpm_plusplus_2M_sampler,
}


class CAD(
    CADRINTextCond,
    PyTorchModelHubMixin,
    repo_url="https://github.com/lucasdegeorge/T2I-ImageNet/",
    pipeline_tag="text-to-image",
    tags=["ImageNet", "diffusion", "arxiv:2502.21318"],
    license="mit",
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_postprocessing():
    vae = AutoencoderKL.from_pretrained(
        "benjamin-paine/stable-diffusion-v1-5", subfolder="vae"
    )
    return SD1_5VAEDecoderPostProcessing(vae)


def scheduler_fn(
    scheduler_type: str, start: float, end: float, tau: float, clip_min: float = 1e-9
):
    if scheduler_type == "sigmoid":
        return SigmoidScheduler(start, end, tau, clip_min)
    elif scheduler_type == "cosine":
        return CosineScheduler(start, end, tau, clip_min)
    elif scheduler_type == "linear":
        return LinearScheduler(clip_min=clip_min)
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")


def load_prepocessing(dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = T5EncoderModel.from_pretrained("google/flan-t5-xl", torch_dtype=dtype)
    return TextPreprocessing(
        tokenizer,
        model,
        drop_labels=False,
        input_key="text",
        output_key_root="text_tokens",
    )


class T2IPipeline:
    """
    The T2IPipeline class is designed to facilitate the generation of images from text prompts using a pre-trained CAD model.
    It integrates various components such as samplers, schedulers, and post-processing techniques to produce high-quality images.

    Initialization:
        T2IPipeline(
            model_path,
            sampler="ddim",
            scheduler="sigmoid",
            postprocessing="sd_1_5_vae",
            scheduler_start=-3,
            scheduler_end=3,
            scheduler_tau=1.1,
            device="cuda",
        )

    Parameters:
        model_path (str): Path to the pre-trained CAD model.
        sampler (str): The sampling method to use. Options are "ddim", "ddpm", "dpm", "dpm_2S", "dpm_2M". Default is "ddim".
        scheduler (str): The scheduler type to use. Options are "sigmoid", "cosine", "linear". Default is "sigmoid".
        postprocessing (str): The post-processing method to use. Options are "consistency-decoder", "sd_1_5_vae". Default is "sd_1_5_vae".
        scheduler_start (float): Start value for the scheduler. Default is -3.
        scheduler_end (float): End value for the scheduler. Default is 3.
        scheduler_tau (float): Tau value for the scheduler. Default is 1.1.
        device (str): Device to run the model on. Default is "cuda".

    Methods:
        model(*args, **kwargs):
            Runs the preconditioning on the network with the provided arguments.

        __call__(...):
            Generates images based on the provided conditions and parameters.

            Parameters:
                cond (str or list of str): The conditioning text or list of texts.
                num_samples (int, optional): Number of samples to generate. If not provided, it is inferred from cond.
                x_N (torch.Tensor, optional): Initial noise tensor. If not provided, it is generated.
                latents (torch.Tensor, optional): Previous latents.
                num_steps (int, optional): Number of steps for the sampler. If not provided, the default is used.
                sampler (callable, optional): Custom sampler function. If not provided, the default sampler is used.
                scheduler (callable, optional): Custom scheduler function. If not provided, the default scheduler is used.
                cfg (float): Classifier-free guidance scale. Default is 15.
                guidance_type (str): Type of guidance. Default is "constant".
                guidance_start_step (int): Step to start guidance. Default is 0.
                generator (torch.Generator, optional): Random number generator.
                coherence_value (float): Doherence value for sampling. Default is 1.0.
                uncoherence_value (float): Uncoherence value for sampling. Default is 0.0.
                unconfident_prompt (str, optional): Unconfident prompt text.
                thresholding_type (str): Type of thresholding. Default is "clamp".
                clamp_value (float): Clamp value for thresholding. Default is 1.0.
                thresholding_percentile (float): Percentile for thresholding. Default is 0.995.

            Returns:
                torch.Tensor: The generated image tensor after post-processing.

        to(device):
            Moves the model and its components to the specified device.

            Parameters:
                device (str): The device to move the model to (e.g., "cuda", "cpu").

            Returns:
                T2IPipeline: The pipeline instance with updated device.

    Example Usage:
            pipe = T2IPipeline(
                "nicolas-dufour/",
            )
            pipe.to("cuda")
            image = pipe(
                "a beautiful landscape with a river and mountains",
                num_samples=4,
            )
    """

    def __init__(
        self,
        model_path,
        sampler="ddim",
        scheduler="linear",
        postprocessing="sd_1_5_vae",
        scheduler_start=1,
        scheduler_end=0,
        scheduler_tau=1,
        device="cuda",
        t5_dtype=torch.float32,
    ):
        self.network = CAD.from_pretrained(model_path)
        self.network.requires_grad_(False).eval()
        assert sampler in SAMPLERS, f"Sampler {sampler} not found"
        assert scheduler in [
            "sigmoid",
            "cosine",
            "linear",
        ], f"Scheduler {scheduler} not supported"
        self.scheduler = scheduler_fn(
            scheduler, scheduler_start, scheduler_end, scheduler_tau
        )
        self.cond_preprocessing = load_prepocessing(t5_dtype)
        self.postprocessing = get_postprocessing()
        self.sampler = SAMPLERS[sampler]
        self.model_path = model_path
        self.preconditioning = DDPMPrecond(
            self.network.num_latents, self.network.latents_dim
        )
        self.device = device

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def __call__(
        self,
        cond,
        num_samples=None,
        x_N=None,
        latents=None,
        num_steps=None,
        sampler=None,
        scheduler=None,
        cfg=15,
        guidance_type="constant",
        guidance_start_step=0,
        generator=None,
        coherence_value=1.0,
        uncoherence_value=0.0,
        unconfident_prompt=None,
        thresholding_type="clamp",
        clamp_value=1.2,
        thresholding_percentile=0.995,
    ):
        batch = {"previous_latents": latents}
        shape = [4, 32, 32]
        if num_samples is None and x_N is None:
            if type(cond) == list:
                num_samples = len(cond)
            else:
                num_samples = 1
        if x_N is None:
            x_N = torch.randn(
                num_samples, *shape, device=self.device, generator=generator
            )
        else:
            x_N = x_N.to(self.device)
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            num_samples = x_N.shape[0]
            shape = x_N.shape[1:]
        batch["y"] = x_N
        if sampler is None:
            sampler = self.sampler
        if scheduler is None:
            scheduler = self.scheduler
        if unconfident_prompt is not None:
            uncond_conditioning = unconfident_prompt
        else:
            uncond_conditioning = generate_negative_prompts(num_samples)
        if isinstance(cond, str):
            cond = [cond] * (1 if num_samples is None else num_samples)
        if isinstance(uncond_conditioning, str):
            uncond_tokens = [uncond_conditioning] * (
                1 if num_samples is None else num_samples
            )
        elif isinstance(uncond_conditioning, list):
            assert len(uncond_conditioning) == num_samples
            uncond_tokens = uncond_conditioning

        batch[self.cond_preprocessing.input_key] = cond
        uncond_tokens_batch = {self.cond_preprocessing.input_key: uncond_tokens}
        uncond_tokens = self.cond_preprocessing(
            uncond_tokens_batch,
            device=self.device,
        )
        batch = self.cond_preprocessing(batch, device=self.device)
        if num_steps is None:
            image = sampler(
                self.model,
                batch,
                conditioning_keys=[self.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                generator=generator,
                use_coherence_sampling=True,
                coherence_value=coherence_value,
                uncoherence_value=uncoherence_value,
                thresholding_type=thresholding_type,
                clamp_value=clamp_value,
                thresholding_percentile=thresholding_percentile,
            )
        else:
            image = sampler(
                self.model,
                batch,
                conditioning_keys=[self.cond_preprocessing.output_key_root],
                scheduler=scheduler,
                uncond_tokens=uncond_tokens,
                num_steps=num_steps,
                cfg_rate=cfg,
                guidance_type=guidance_type,
                guidance_start_step=guidance_start_step,
                generator=generator,
                use_coherence_sampling=True,
                coherence_value=coherence_value,
                uncoherence_value=uncoherence_value,
                thresholding_type=thresholding_type,
                clamp_value=clamp_value,
                thresholding_percentile=thresholding_percentile,
            )
        return self.postprocessing(image)

    def to(self, device):
        self.network.to(device)
        self.cond_preprocessing.to(device)
        self.postprocessing.to(device)
        self.device = torch.device(device)
        return self


def generate_random_string():
    words = []
    for _ in range(random.randint(3, 6)):
        word = "".join(
            random.choice(string.ascii_lowercase) for _ in range(random.randint(3, 7))
        )
        words.append(word)
    return " ".join(words)


def generate_negative_prompts(batch_size):
    return [generate_random_string() for _ in range(batch_size)]
