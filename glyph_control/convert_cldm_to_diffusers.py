import os
import shutil
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import gradio as gr
import requests
import torch
from huggingface_hub import Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_controlnet_from_original_ckpt,
    download_from_original_stable_diffusion_ckpt,
)


COMMIT_MESSAGE = (
    " This PR adds fp32 and fp16 weights in PyTorch and safetensors format to {}"
)


def convert_single(
    model_id: str,
    token: str,
    filename: str,
    model_type: str,
    sample_size: int,
    scheduler_type: str,
    extract_ema: bool,
    folder: str,
    progress,
    subfolder=None,
):
    from_safetensors = filename.endswith(".safetensors")

    progress(0, desc="Downloading model")
    local_file = os.path.join(model_id, filename)
    ckpt_file = (
        local_file
        if os.path.isfile(local_file)
        else hf_hub_download(
            repo_id="AIGText/GlyphControl",
            repo_type="space",
            filename=filename,
            token=token,
            subfolder=subfolder,
        )
    )

    if model_type == "v1":
        config_url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    elif model_type == "v2":
        if sample_size == 512:
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml"
        else:
            config_url = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    elif model_type == "ControlNet":
        config_url = (Path(model_id) / "resolve/main" / filename).with_suffix(".yaml")
        config_url = "https://huggingface.co/" + str(config_url)
    elif model_type == "GlyphControl":
        config_url = "https://github.com/AIGText/GlyphControl-release/raw/main/configs/config.yaml"

    # config_file = BytesIO(requests.get(config_url).content)

    response = requests.get(config_url)
    with tempfile.NamedTemporaryFile(delete=False, mode="wb", dir="./") as tmp_file:
        tmp_file.write(response.content)
        temp_config_file_path = tmp_file.name

    if model_type == "ControlNet":
        progress(0.2, desc="Converting ControlNet Model")
        pipeline = download_controlnet_from_original_ckpt(
            ckpt_file,
            temp_config_file_path,
            image_size=sample_size,
            from_safetensors=from_safetensors,
            extract_ema=extract_ema,
        )
        to_args = {"dtype": torch.float16}
    elif model_type == "GlyphControl":
        progress(0.1, desc="Converting Model")
        pipeline = download_controlnet_from_original_ckpt(
            ckpt_file,
            temp_config_file_path,
            image_size=sample_size,
            from_safetensors=from_safetensors,
            extract_ema=extract_ema,
        )
        to_args = {"dtype": torch.float16}
    else:
        progress(0.1, desc="Converting Model")
        pipeline = download_from_original_stable_diffusion_ckpt(
            ckpt_file,
            temp_config_file_path,
            image_size=sample_size,
            scheduler_type=scheduler_type,
            from_safetensors=from_safetensors,
            extract_ema=extract_ema,
        )
        to_args = {"dtype": torch.float16}

    pipeline.save_pretrained(folder)
    pipeline.save_pretrained(folder, safe_serialization=True)

    pipeline = pipeline.to(**to_args)
    pipeline.save_pretrained(folder, variant="fp16")
    pipeline.save_pretrained(folder, safe_serialization=True, variant="fp16")

    return folder


def previous_pr(api: "HfApi", model_id: str, pr_title: str) -> Optional["Discussion"]:
    try:
        discussions = api.get_repo_discussions(repo_id=model_id)
    except Exception:
        return None
    for discussion in discussions:
        if (
            discussion.status == "open"
            and discussion.is_pull_request
            and discussion.title == pr_title
        ):
            details = api.get_discussion_details(
                repo_id=model_id, discussion_num=discussion.num
            )
            if details.target_branch == "refs/heads/main":
                return discussion


def convert(
    token: str,
    model_id: str,
    filename: str,
    model_type: str,
    sample_size: int = 512,
    scheduler_type: str = "pndm",
    extract_ema: bool = True,
    revision: Optional[str] = None,
    progress=gr.Progress(),
):
    api = HfApi()

    pr_title = "Adding `diffusers` weights of this model"

    with TemporaryDirectory() as d:
        folder = os.path.join(d, repo_folder_name(repo_id=model_id, repo_type="models"))
        os.makedirs(folder)
        new_pr = None
        try:
            folder = convert_single(
                model_id,
                token,
                filename,
                model_type,
                sample_size,
                scheduler_type,
                extract_ema,
                folder,
                progress,
                subfolder="checkpoints",
            )
            progress(0.7, desc="Uploading to Hub")
            new_pr = api.upload_folder(
                folder_path=folder,
                path_in_repo="./",
                repo_id=model_id,
                repo_type="model",
                token=token,
                commit_message=filename.split(".")[0],
                commit_description=COMMIT_MESSAGE.format(model_id),
                # create_pr=True,
                revision=revision
            )
            pr_number = new_pr.split("%2F")[-1].split("/")[0]
            link = f"Pr created at: {'https://huggingface.co/' + os.path.join(model_id, 'discussions', pr_number)}"
            progress(1, desc="Done")
        except Exception as e:
            raise gr.exceptions.Error(str(e))
        finally:
            shutil.rmtree(folder)

        return link
