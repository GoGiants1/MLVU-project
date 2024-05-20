# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import accelerate
import cv2
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import disable_caching
from huggingface_hub import HfFolder, Repository, create_repo, hf_hub_download, whoami
from model.layout_generator import get_layout_from_prompt
from model.text_segmenter.unet import UNet
from packaging import version
from PIL import (
    Image,
    ImageEnhance,
    ImageOps,
)
from safetensors import safe_open
from t_diffusers.scheduling_ddpm import DDPMScheduler
from t_diffusers.unet_2d_condition import UNet2DConditionModel

# import for visualization
from termcolor import colored
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from util import (
    combine_image,
    filter_segmentation_mask,
    make_caption_pil,
    segmentation_mask_visualization,
    transform_mask,
)

import diffusers
from diffusers import AutoencoderKL
from diffusers.models import ImageProjection
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, _get_model_file
from diffusers.utils import check_min_version


"""
from diffusers.utils.import_utils import is_xformers_available
"""

disable_caching()
check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",  # no need to modify this
        help="Path to pretrained model or model identifier from huggingface.co/models. Please do not modify this.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        required=True,
        choices=["text-to-image", "text-to-image-with-template", "text-inpainting"],
        help="Three modes can be used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        required=True,
        help="The text prompts provided by users.",
    )
    parser.add_argument(
        "--template_image",
        type=str,
        default="",
        help="The template image should be given when using 【text-to-image-with-template】 mode.",
    )
    parser.add_argument(
        "--original_image",
        type=str,
        default="",
        help="The original image should be given when using 【text-inpainting】 mode.",
    )
    parser.add_argument(
        "--text_mask",
        type=str,
        default="",
        help="The text mask should be given when using 【text-inpainting】 mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--classifier_free_scale",
        type=float,
        default=7.5,  # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Classifier free scale following https://arxiv.org/abs/2207.12598.",
    )
    parser.add_argument(
        "--drop_caption",
        action="store_true",
        help="Whether to drop captions during training following https://arxiv.org/abs/2207.12598..",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="GoGiants1/td-unet15",  # should be specified during inference
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="assets/font/Arial.ttf",
        help="The path of font for visualization.",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,  # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Diffusion steps for sampling.",
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=9,  # please decreases the number if out-of-memory error occurs
        help="Number of images to be sample. Please decrease it when encountering out of memory error.",
    )
    parser.add_argument(
        "--binarization",
        action="store_true",
        help="Whether to binarize the template image.",
    )
    parser.add_argument(
        "--use_pillow_segmentation_mask",
        type=bool,
        default=True,
        help="In the 【text-to-image】 mode, please specify whether to use the segmentation masks provided by PILLOW",
    )
    parser.add_argument(
        "--character_segmenter_path",
        type=str,
        default="textdiffuser-ckpt/text_segmenter.pth",
        help="checkpoint of character-level segmenter",
    )
    args = parser.parse_args()

    print(f'{colored("[√]", "green")} Arguments are loaded.')
    print(args)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def load_ip_adapter(
    pretrained_model_name_or_path_or_dict: Union[
        str, List[str], Dict[str, torch.Tensor]
    ],
    subfolder: Union[str, List[str]],
    weight_name: Union[str, List[str]],
    image_encoder_folder: Optional[str] = "image_encoder",
    **kwargs,
):
    """
    Parameters:
        pretrained_model_name_or_path_or_dict (`str` or `List[str]` or `os.PathLike` or `List[os.PathLike]` or `dict` or `List[dict]`):
            Can be either:

                - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                  the Hub.
                - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                  with [`ModelMixin.save_pretrained`].
                - A [torch state
                  dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
        subfolder (`str` or `List[str]`):
            The subfolder location of a model file within a larger model repository on the Hub or locally.
            If a list is passed, it should have the same length as `weight_name`.
        weight_name (`str` or `List[str]`):
            The name of the weight file to load. If a list is passed, it should have the same length as
            `weight_name`.
        image_encoder_folder (`str`, *optional*, defaults to `image_encoder`):
            The subfolder location of the image encoder within a larger model repository on the Hub or locally.
            Pass `None` to not load the image encoder. If the image encoder is located in a folder inside `subfolder`,
            you only need to pass the name of the folder that contains image encoder weights, e.g. `image_encoder_folder="image_encoder"`.
            If the image encoder is located in a folder other than `subfolder`, you should pass the path to the folder that contains image encoder weights,
            for example, `image_encoder_folder="different_subfolder/image_encoder"`.
    """

    # handle the list inputs for multiple IP Adapters
    if not isinstance(weight_name, list):
        weight_name = [weight_name]

    if not isinstance(pretrained_model_name_or_path_or_dict, list):
        pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
    if len(pretrained_model_name_or_path_or_dict) == 1:
        pretrained_model_name_or_path_or_dict = (
            pretrained_model_name_or_path_or_dict * len(weight_name)
        )

    if not isinstance(subfolder, list):
        subfolder = [subfolder]
    if len(subfolder) == 1:
        subfolder = subfolder * len(weight_name)

    if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
        raise ValueError(
            "`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length."
        )

    if len(weight_name) != len(subfolder):
        raise ValueError("`weight_name` and `subfolder` must have the same length.")

    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }
    state_dicts = []
    for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder
    ):
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = (
                                f.get_tensor(key)
                            )
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = (
                                f.get_tensor(key)
                            )
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["image_proj", "ip_adapter"]:
            raise ValueError(
                "Required keys are (`image_proj` and `ip_adapter`) missing from the state dict."
            )

        state_dicts.append(state_dict)

        # load CLIP image encoder here if it has not been registered to the pipeline yet

        if image_encoder_folder is not None:
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                logger.info(
                    f"loading image_encoder from {pretrained_model_name_or_path_or_dict}"
                )
                if image_encoder_folder.count("/") == 0:
                    image_encoder_subfolder = Path(
                        subfolder, image_encoder_folder
                    ).as_posix()
                else:
                    image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    pretrained_model_name_or_path_or_dict,
                    subfolder=image_encoder_subfolder,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
            else:
                raise ValueError(
                    "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                )
        else:
            logger.warning(
                "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
            )

    # create feature extractor if it has not been registered to the pipeline yet

    feature_extractor = CLIPImageProcessor()

    # load ip-adapter into unet
    # unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
    # unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)

    return state_dicts, feature_extractor, image_encoder


def encode_image(
    image_encoder: CLIPVisionModelWithProjection,  # clip encoder
    feature_extractor: CLIPImageProcessor,
    image,
    device,
    num_images_per_prompt,
    output_hidden_states=None,
):
    dtype = next(image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = feature_extractor(
            image, return_tensors="pt"
        ).pixel_values  # torch가 아니면 torch로 변환

    image = image.to(device=device, dtype=dtype)
    if output_hidden_states:
        image_enc_hidden_states = image_encoder(
            image, output_hidden_states=True
        ).hidden_states[-2]
        image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        uncond_image_enc_hidden_states = image_encoder(
            torch.zeros_like(image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_enc_hidden_states = (
            uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
        )
        return image_enc_hidden_states, uncond_image_enc_hidden_states
    else:
        image_embeds = image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds


def prepare_ip_adapter_image_embeds(
    unet,
    image_encoder,
    feature_extractor,
    ip_adapter_image,  # masked_image
    ip_adapter_image_embeds,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
):
    if ip_adapter_image_embeds is None:
        if not isinstance(ip_adapter_image, list):
            ip_adapter_image = [ip_adapter_image]

        if len(ip_adapter_image) != len(unet.encoder_hid_proj.image_projection_layers):
            raise ValueError(
                f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
            )

        image_embeds = []
        for single_ip_adapter_image, image_proj_layer in zip(
            ip_adapter_image, unet.encoder_hid_proj.image_projection_layers
        ):
            output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
            single_image_embeds, single_negative_image_embeds = encode_image(
                image_encoder,
                feature_extractor,
                single_ip_adapter_image,
                device,
                1,
                output_hidden_state,
            )
            single_image_embeds = torch.stack(
                [single_image_embeds] * num_images_per_prompt, dim=0
            )
            single_negative_image_embeds = torch.stack(
                [single_negative_image_embeds] * num_images_per_prompt, dim=0
            )

            if do_classifier_free_guidance:
                single_image_embeds = torch.cat(
                    [single_negative_image_embeds, single_image_embeds]
                )
                single_image_embeds = single_image_embeds.to(device)

            image_embeds.append(single_image_embeds)
    else:
        repeat_dims = [1]
        image_embeds = []
        for single_image_embeds in ip_adapter_image_embeds:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = (
                    single_image_embeds.chunk(2)
                )
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt,
                    *(repeat_dims * len(single_image_embeds.shape[1:])),
                )
                single_negative_image_embeds = single_negative_image_embeds.repeat(
                    num_images_per_prompt,
                    *(repeat_dims * len(single_negative_image_embeds.shape[1:])),
                )
                single_image_embeds = torch.cat(
                    [single_negative_image_embeds, single_image_embeds]
                )
            else:
                single_image_embeds = single_image_embeds.repeat(
                    num_images_per_prompt,
                    *(repeat_dims * len(single_image_embeds.shape[1:])),
                )
            image_embeds.append(single_image_embeds)

    return image_embeds


# @torchsnooper.snoop()
def main():
    args = parse_args()
    # If passed along, set the training seed now.
    seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    set_seed(seed)
    print(f'{colored("[√]", "green")} Seed is set to {seed}.')

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    sub_output_dir = f"{args.prompt}_[{args.mode.upper()}]_[SEED-{seed}]"

    print(f'{colored("[√]", "green")} Logging dir is set to {logging_dir}.')

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            _ = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            print(args.output_dir)

    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # TODO: AnyText Ideas! -> use multi-lingual BERT
    # from transformers import BertTokenizer, BertModel
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # model = BertModel.from_pretrained("bert-base-multilingual-cased")
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).cuda()
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.resume_from_checkpoint, subfolder="unet", revision=None
    ).cuda()

    # load ip-adapter
    # from https://huggingface.co/h94/IP-Adapter/tree/main/models

    state_dicts, feature_extractor, image_encoder = load_ip_adapter(
        "h94/IP-Adapter",
        subfolder=[
            "models",
        ],
        weight_name=[
            "ip-adapter_sd15.safetensors",
        ],
        image_encoder_folder="image_encoder",
    )

    unet._load_ip_adapter_weights(
        state_dicts, low_cpu_mem_usage=_LOW_CPU_MEM_USAGE_DEFAULT
    )
    print(f'{colored("[√]", "green")} load ip_adapter into unet')
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    ### Newly Added
    # feature_extractor.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    # feature_extractor
    image_encoder.to("cuda", torch.float32)
    ###

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # setup schedulers
    scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler.set_timesteps(args.sample_steps)
    sample_num = args.vis_num
    noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
    input = noise  # (b, 4, 64, 64)
    scene_caption = args.prompt.split("'")[0]+args.prompt.split("'")[2]
    captions = [scene_caption] * sample_num
    captions_nocond = [""] * sample_num
    print(f'{colored("[√]", "green")} Prompt is loaded: {args.prompt}.')

    # encode text prompts
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids  # (b, 77) id로 token을 숫자화
    encoder_hidden_states = text_encoder(inputs)[0].cuda()  # (b, 77, 768)
    print(
        f'{colored("[√]", "green")} encoder_hidden_states: {encoder_hidden_states.shape}.'
    )

    inputs_nocond = tokenizer(
        captions_nocond,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids  # (b, 77)
    encoder_hidden_states_nocond = text_encoder(inputs_nocond)[
        0
    ].cuda()  # (b, 77, 768) 1024size로 나중에 rescale?
    print(
        f'{colored("[√]", "green")} encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.'
    )

    character_segmenter_path = hf_hub_download(
        "GoGiants1/td-unet15", "text_segmenter.pth", repo_type="model"
    )
    print(f'{colored("[√]", "green")} character_segmenter_path downloaded.')

    # load character-level segmenter
    segmenter = UNet(3, 96, True).cuda()
    segmenter = torch.nn.DataParallel(segmenter)
    segmenter.load_state_dict(torch.load(character_segmenter_path))
    segmenter.eval()
    print(f'{colored("[√]", "green")} Text segmenter is successfully loaded.')
    image = Image.open(args.original_image).convert("RGB").resize((512, 512))
    print(image, image.size)
    ip_adapter_image = image

    batch_size = args.vis_num
    num_images_per_prompt = 1

    image_embeds = prepare_ip_adapter_image_embeds(
        unet,
        image_encoder,
        feature_extractor,
        ip_adapter_image,
        None,
        "cuda",
        batch_size * num_images_per_prompt,
        True,
    )

    added_cond_kwargs_cond = (
        {"image_embeds": image_embeds} if (ip_adapter_image is not None) else None
    )



    #### text-to-image ####
    if args.mode == "text-to-image":
        render_image, segmentation_mask_from_pillow = get_layout_from_prompt(args)

        if args.use_pillow_segmentation_mask:
            segmentation_mask = torch.Tensor(
                np.array(segmentation_mask_from_pillow)
            ).cuda()  # (512, 512)
        else:
            to_tensor = transforms.ToTensor()
            image_tensor = (
                to_tensor(render_image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
            )
            with torch.no_grad():
                segmentation_mask = segmenter(image_tensor)
            segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)

        segmentation_mask = filter_segmentation_mask(segmentation_mask)
        segmentation_mask = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(256, 256),
            mode="nearest",
        )
        segmentation_mask = (
            segmentation_mask.squeeze(1).repeat(sample_num, 1, 1).long().to("cuda")
        )  # (1, 1, 256, 256)
        print(
            f'{colored("[√]", "green")} character-level segmentation_mask: {segmentation_mask.shape}.'
        )

        feature_mask = torch.ones(sample_num, 1, 64, 64).to("cuda")  # (b, 1, 64, 64)
        masked_image = torch.zeros(sample_num, 3, 512, 512).to(
            "cuda"
        )  # (b, 3, 512, 512)
        masked_feature = vae.encode(masked_image).latent_dist.sample()  # (b, 4, 64, 64)
        masked_feature = masked_feature * vae.config.scaling_factor
        print(f'{colored("[√]", "green")} feature_mask: {feature_mask.shape}.')
        print(f'{colored("[√]", "green")} masked_feature: {masked_feature.shape}.')

    #### text-to-image-with-template ####
    if args.mode == "text-to-image-with-template":
        template_image = (
            Image.open(args.template_image).resize((256, 256)).convert("RGB")
        )

        # whether binarization is needed
        print(
            f'{colored("[Warning]", "red")} args.binarization is set to {args.binarization}. You may need it when using handwritten images as templates.'
        )
        if args.binarization:
            gray = ImageOps.grayscale(template_image)
            binary = gray.point(lambda x: 255 if x > 96 else 0, "1")
            template_image = binary.convert("RGB")

        to_tensor = transforms.ToTensor()
        image_tensor = (
            to_tensor(template_image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
        )  # (b, 3, 256, 256)

        with torch.no_grad():
            segmentation_mask = segmenter(image_tensor)  # (b, 96, 256, 256)
        segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)  # (256, 256)
        segmentation_mask = filter_segmentation_mask(segmentation_mask)  # (256, 256)
        segmentation_mask_pil = Image.fromarray(
            segmentation_mask.type(torch.uint8).cpu().numpy()
        ).convert("RGB")

        segmentation_mask = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(256, 256),
            mode="nearest",
        )  # (b, 1, 256, 256)
        segmentation_mask = (
            segmentation_mask.squeeze(1).repeat(sample_num, 1, 1).long().to("cuda")
        )  # (b, 1, 256, 256)
        print(
            f'{colored("[√]", "green")} Character-level segmentation_mask: {segmentation_mask.shape}.'
        )

        feature_mask = torch.ones(sample_num, 1, 64, 64).to("cuda")  # (b, 1, 64, 64)
        masked_image = torch.zeros(sample_num, 3, 512, 512).to(
            "cuda"
        )  # (b, 3, 512, 512)
        masked_feature = vae.encode(masked_image).latent_dist.sample()  # (b, 4, 64, 64)
        masked_feature = masked_feature * vae.config.scaling_factor  # (b, 4, 64, 64)
        print(f'{colored("[√]", "green")} feature_mask: {feature_mask.shape}.')
        print(f'{colored("[√]", "green")} masked_feature: {masked_feature.shape}.')

        render_image = template_image  # for visualization

    #### text-inpainting ####
    if args.mode == "text-inpainting":
        text_mask = cv2.imread(args.text_mask)
        threshold = 50
        _, text_mask = cv2.threshold(text_mask, threshold, 255, cv2.THRESH_BINARY)
        text_mask = Image.fromarray(text_mask).convert("RGB").resize((256, 256))
        text_mask_tensor = (
            transforms.ToTensor()(text_mask).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
        )
        with torch.no_grad():
            segmentation_mask = segmenter(text_mask_tensor)

        segmentation_mask = segmentation_mask.max(1)[1].squeeze(0)
        segmentation_mask = filter_segmentation_mask(segmentation_mask)
        segmentation_mask = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(),
            size=(256, 256),
            mode="nearest",
        )

        image_mask = transform_mask(args.text_mask)
        image_mask = torch.from_numpy(image_mask).cuda().unsqueeze(0).unsqueeze(0)

        image = Image.open(args.original_image).convert("RGB").resize((512, 512))
        image_tensor = (
            transforms.ToTensor()(image).unsqueeze(0).cuda().sub_(0.5).div_(0.5)
        )
        # masked_image = image_tensor * (1 - image_mask)
        # masked_feature = (
        #     vae.encode(masked_image).latent_dist.sample().repeat(sample_num, 1, 1, 1)
        # )
        # masked_feature = masked_feature * vae.config.scaling_factor
        masked_image = torch.zeros(sample_num, 3, 512, 512).to(
            "cuda"
        )  # (b, 3, 512, 512)
        masked_feature = vae.encode(masked_image).latent_dist.sample()  # (b, 4, 64, 64)
        masked_feature = masked_feature * vae.config.scaling_factor

        image_mask = torch.nn.functional.interpolate(
            image_mask, size=(256, 256), mode="nearest"
        ).repeat(sample_num, 1, 1, 1)
        segmentation_mask = segmentation_mask * image_mask
        # feature_mask = torch.nn.functional.interpolate(
        #     image_mask, size=(64, 64), mode="nearest"
        # )
        feature_mask = torch.ones(sample_num, 1, 64, 64).to("cuda")  # (b, 1, 64, 64)

        print(f'{colored("[√]", "green")} feature_mask: {feature_mask.shape}.')
        print(
            f'{colored("[√]", "green")} segmentation_mask: {segmentation_mask.shape}.'
        )
        print(f'{colored("[√]", "green")} masked_feature: {masked_feature.shape}.')

        render_image = Image.open(args.original_image)

    # diffusion process
    intermediate_images = []

    feature_mask = torch.cat([feature_mask]*2)
    masked_feature = torch.cat([masked_feature]*2)
    segmentation_mask = torch.cat([segmentation_mask]*2)
    encoder_hidden_states = torch.cat([encoder_hidden_states,encoder_hidden_states_nocond])
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            latent_model_input = torch.cat([input]*2)
            latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t
                )
            noise_pred = unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                segmentation_mask=segmentation_mask,
                feature_mask=feature_mask,
                masked_feature=masked_feature,
                added_cond_kwargs=added_cond_kwargs_cond,  # Added for IP-Adapter
            ).sample  # b, 4, 64, 64

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noisy_residual = noise_pred_uncond + args.classifier_free_scale * (
                noise_pred_cond - noise_pred_uncond
            )  # b, 4, 64, 64
            input = scheduler.step(noisy_residual, t, input, return_dict=False)[0]
            intermediate_images.append(input)

    # decode and visualization
    input = 1 / vae.config.scaling_factor * input
    sample_images = vae.decode(input.float(), return_dict=False)[0]  # (b, 3, 512, 512)

    image_pil = render_image.resize((512, 512))
    segmentation_mask = segmentation_mask[0].squeeze().cpu().numpy()
    character_mask_pil = Image.fromarray(
        ((segmentation_mask != 0) * 255).astype("uint8")
    ).resize((512, 512))
    character_mask_highlight_pil = segmentation_mask_visualization(
        args.font_path, segmentation_mask
    )
    caption_pil = make_caption_pil(args.font_path, captions)

    # save pred_img
    pred_image_list = []
    for image in sample_images.float():
        image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8")).convert("RGB")
        pred_image_list.append(image)

    os.makedirs(f"{args.output_dir}/{sub_output_dir}", exist_ok=True)

    # save additional info
    if args.mode == "text-to-image":
        image_pil.save(
            os.path.join(args.output_dir, sub_output_dir, "render_text_image.png")
        )
        enhancer = ImageEnhance.Brightness(segmentation_mask_from_pillow)
        im_brightness = enhancer.enhance(5)
        im_brightness.save(
            os.path.join(
                args.output_dir, sub_output_dir, "segmentation_mask_from_pillow.png"
            )
        )
    if args.mode == "text-to-image-with-template":
        template_image.save(
            os.path.join(args.output_dir, sub_output_dir, "template.png")
        )
        enhancer = ImageEnhance.Brightness(segmentation_mask_pil)
        im_brightness = enhancer.enhance(5)
        im_brightness.save(
            os.path.join(
                args.output_dir, sub_output_dir, "segmentation_mask_from_template.png"
            )
        )
    if args.mode == "text-inpainting":
        character_mask_highlight_pil = character_mask_pil
        # background
        background = Image.open(args.original_image).resize((512, 512))
        alpha = Image.new("L", background.size, int(255 * 0.2))
        background.putalpha(alpha)
        # foreground
        foreground = Image.open(args.text_mask).convert("L").resize((512, 512))
        threshold = 200
        alpha = foreground.point(lambda x: 0 if x > threshold else 255, "1")
        foreground.putalpha(alpha)
        character_mask_pil = Image.alpha_composite(
            foreground.convert("RGBA"), background.convert("RGBA")
        ).convert("RGB")
        # merge
        pred_image_list_new = []
        for pred_image in pred_image_list:
            '''
            pred_image = inpainting_merge_image(
                Image.open(args.original_image),
                Image.open(args.text_mask).convert("L"),
                pred_image,
            )
            '''
            pred_image_list_new.append(pred_image)
        pred_image_list = pred_image_list_new

    combine_image(
        args,
        sub_output_dir,
        pred_image_list,
        image_pil,
        character_mask_pil,
        character_mask_highlight_pil,
        caption_pil,
    )

    # create a soft link
    if os.path.exists(os.path.join(args.output_dir, "latest")):
        os.unlink(os.path.join(args.output_dir, "latest"))
    os.symlink(
        os.path.abspath(os.path.join(args.output_dir, sub_output_dir)),
        os.path.abspath(os.path.join(args.output_dir, "latest/")),
    )

    color_sub_output_dir = colored(sub_output_dir, "green")
    print(
        f'{colored("[√]", "green")} Save successfully. Please check the output at {color_sub_output_dir} OR the latest folder'
    )


if __name__ == "__main__":
    main()
