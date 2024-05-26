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
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import disable_caching
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from model.layout_generator import get_layout_from_prompt
from model.text_segmenter.unet import UNet
from packaging import version
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'eval')))

from clipscore.clipscore import cal_clip_score
from fid.src.pytorch_fid.fid_score import cal_fid
from ocr import ocr_eval

# import for visualization
from termcolor import colored
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from util import (
    filter_segmentation_mask,
    make_caption_pil,
    segmentation_mask_visualization,
)

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    
)
from t_diffusers.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available


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
        required=False,
        help="The text prompts provided by users.",
    )
    parser.add_argument(
        "--prompt_list",
        type=str,
        default="",
        required=True,
        help="The list of prompts.",
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
        help="The path of the generation directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,  # set to 0 during evaluation
        help="A seed for reproducible training.",
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
        default=None,  # should be specified during inference
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--target_images",
        type=str,
        default=None,  
        required=True,
        help=(
            "prompt 리스트의 n번째 줄과 내용이 맞는 이미지들은 n번째 폴더 즉 이름이 숫자 n인 폴더에 넣는다, target_images 폴더안에 이름이 n인폴더존재"
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
        default="Arial.ttf",
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


# @torchsnooper.snoop()
def main():
    args = parse_args()
    # If passed along, set the training seed now.
    seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    set_seed(seed)
    print(f'{colored("[√]", "green")} Seed is set to {seed}.')

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    # sub_output_dir = f"{args.prompt}_[{args.mode.upper()}]_[SEED-{seed}]"

    print(f'{colored("[√]", "green")} Logging dir is set to {logging_dir}.')

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
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
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token
            )

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
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).cuda()
    
    unet = UNet2DConditionModel.from_pretrained(
        args.resume_from_checkpoint, subfolder="unet", revision=None
    ).cuda()

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

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

    with open(args.prompt_list) as fr:
        prompts = fr.readlines()
        prompts = [_.strip() for _ in prompts]

    for idx in range(args.vis_num):
        os.makedirs(
            os.path.join(args.output_dir, "textdiffuser", "images_" + str(idx)),
            exist_ok=True,
        )

    for prompt_index, prompt in enumerate(prompts):

        args.prompt = prompt

        # setup schedulers
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        scheduler.set_timesteps(args.sample_steps)
        sample_num = args.vis_num
        noise = torch.randn((sample_num, 4, 64, 64)).to("cuda")  # (b, 4, 64, 64)
        input = noise  # (b, 4, 64, 64)

        captions = [args.prompt] * sample_num
        captions_nocond = [""] * sample_num
        print(f'{colored("[√]", "green")} Prompt is loaded: {args.prompt}.')

        # encode text prompts
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids  # (b, 77)
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
        ].cuda()  # (b, 77, 768)
        print(
            f'{colored("[√]", "green")} encoder_hidden_states_nocond: {encoder_hidden_states_nocond.shape}.'
        )

        # load character-level segmenter
        segmenter = UNet(3, 96, True).cuda()
        segmenter = torch.nn.DataParallel(segmenter)
        segmenter.load_state_dict(torch.load(args.character_segmenter_path))
        segmenter.eval()
        print(f'{colored("[√]", "green")} Text segmenter is successfully loaded.')

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

            feature_mask = torch.ones(sample_num, 1, 64, 64).to(
                "cuda"
            )  # (b, 1, 64, 64)
            masked_image = torch.zeros(sample_num, 3, 512, 512).to(
                "cuda"
            )  # (b, 3, 512, 512)
            masked_feature = vae.encode(
                masked_image
            ).latent_dist.sample()  # (b, 4, 64, 64)
            masked_feature = masked_feature * vae.config.scaling_factor
            print(f'{colored("[√]", "green")} feature_mask: {feature_mask.shape}.')
            print(f'{colored("[√]", "green")} masked_feature: {masked_feature.shape}.')

        # diffusion process
        intermediate_images = []
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():
                noise_pred_cond = unet(
                    sample=input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature,
                ).sample  # b, 4, 64, 64
                noise_pred_uncond = unet(
                    sample=input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states_nocond,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature,
                ).sample  # b, 4, 64, 64
                noisy_residual = noise_pred_uncond + args.classifier_free_scale * (
                    noise_pred_cond - noise_pred_uncond
                )  # b, 4, 64, 64
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample
                intermediate_images.append(prev_noisy_sample)

        # decode and visualization
        input = 1 / vae.config.scaling_factor * input
        sample_images = vae.decode(input.float(), return_dict=False)[
            0
        ]  
        
        # (b, 3, 512, 512)
        #print(sample_images.shape)
        print(captions)
        
        #시작
        
        
        
        #clip score 계산
         
        sample_images2=torch.permute(sample_images,(0,2,3,1)).cpu()
        sample_images2=np.array(sample_images2)
        clip_score=cal_clip_score(sample_images2,captions)
        
        print("\n")
        print(f"Aveage CLIP SCORE of the prompt **{captions[0]}** : ",clip_score)
        print("\n")
        
        #fid score 계산 

        import glob 
        target_images=glob.glob(f"{args.target_images}/{prompt_index+1}/*")
        target_list=[]
        
        w,h=sample_images2.shape[1],sample_images2.shape[2]
        while len(target_list)<sample_images2.shape[0]:

            for img in target_images:
                if len(target_list)==sample_images2.shape[0]:
                    break
            
                try:
                    img=Image.open(img)
                    img=img.resize((w,h))
                    img=np.array(img)
                    target_list.append(img)
                except:
                    continue
            
        target_images2 = np.stack(target_list)
        assert sample_images2.shape==target_images2.shape

        final_fid=cal_fid(sample_images2,target_images2)

        print("\n")
        print(f"FID SCORE of the prompt **{captions[0]}** : ",final_fid)
        print("\n")
        
        #ocr score 계산 
        print("\n")
        a,b,c=ocr_eval(sample_images2,captions[0])
        
        print("OCR EM Score, 대문자 소문자 구별")
        print(a)
        print("OCR EM WC Score, 대문자 소문자 구별 안함")
        print(b)
        print("OCR LEV")
        print(c)
        print("\n")
        
        #끝
        
        

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
            image = Image.fromarray((image * 255).round().astype("uint8")).convert(
                "RGB"
            )
            pred_image_list.append(image)

        for image_index, image in enumerate(pred_image_list):
            image.save(
                f"{args.output_dir}/textdiffuser/images_{image_index}/{prompt_index}_{image_index}.jpg"
            )
            
        print("One prompt done")
        


if __name__ == "__main__":
    main()
