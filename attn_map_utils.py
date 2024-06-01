import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


attn_maps = defaultdict(list)
ip_attn_maps = defaultdict(list)


def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            # print("hook_fn: ", name, "inference_step: ", module.processor.inference_step)
            print(module.processor.attn_map.shape)
            map = module.processor.attn_map.detach().cpu()
            attn_maps[name].append(map)  # 리스트가 아닌 텐서가 들어갈 것임.

            del module.processor.attn_map

        if hasattr(module.processor, "ip_attn_map"):
            print("hook_fn: ", name)
            # dict{inference_step: ip_attn_map_list}
            # downsample 해서 메모리 관리하기 현재 맵 차원 ()
            ip_attn_maps[name].extend(
                module.processor.ip_attn_map
            )  # 리스트가 들어갈 것

            del module.processor.ip_attn_map
            module.processor.ip_attn_map = []

    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        # print(name)
        if name.split(".")[-1].startswith("attn2"):  # attn2에서 attn으로 변경해둔 상태.
            # print("register hook: ", name)
            module.register_forward_hook(hook_fn(name))

    return unet


def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)  # (C, W, H) -> (W, H)
    print("attn_map.shape after mean op: ", attn_map.shape)
    attn_map = attn_map.permute(1, 0)  # (W, H) -> (H, W)
    temp_size = None

    for i in range(0, 5):

        scale = 2**i
        print(
            "scale: ",
            scale,
            "attn_map.shape: ",
            attn_map.shape,
            "temp_w: ",
            target_size[0] // scale,
            "temp_h",
            target_size[1] // scale,
        )
        if (target_size[0] // scale) * (target_size[1] // scale) == attn_map.shape[
            1
        ] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break
    if temp_size is None:
        # target size가 작을 경우 대응
        temp_size = [1, 1 * target_size[1] // target_size[0]]

        while temp_size[0] * temp_size[1] < attn_map.shape[1]:
            temp_size = (temp_size[0] * 2, temp_size[1] * 2)

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)  # (H, W) -> (C, W, H)
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    print("reshaped attn_map.shape: ", attn_map.shape)

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map


def upscale_maps(attn_maps, target_size):
    attn_maps = attn_maps.permute(1, 0)
    temp_size = None

    for i in range(0, 5):
        scale = 2**i
        if (target_size[0] // scale) * (target_size[1] // scale) == attn_maps.shape[
            1
        ] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_maps = F.interpolate(
        attn_maps.to(dtype=torch.float32),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    attn_maps = torch.softmax(attn_maps, dim=1)
    return attn_maps


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size)
        net_attn_maps.append(attn_map)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)

    return net_attn_maps


def get_net_attn_map_per_epochs(
    image_size,
    batch_size=2,
    instance_or_negative=False,
    detach=True,
    target_processor="ip_attn",
):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = defaultdict(list)
    target_attn_map_dict = attn_maps if target_processor == "attn" else ip_attn_maps

    # print("target_attn_map_dict: ", target_attn_map_dict.keys())
    if target_processor == "ip_attn":
        for name, attn_map_list in target_attn_map_dict.items():
            print(name)
            if not attn_map_list:
                continue

            for attn_map in attn_map_list:
                attn_map_1 = attn_map[0].cpu() if detach else attn_map[0]
                attn_map_2 = attn_map[1].cpu() if detach else attn_map[1]
                attn_map_1 = torch.chunk(attn_map_1, batch_size)[idx]
                # .squeeze() # chunk의 첫번째가 bbox 마스크, 두번째가 tss 마스크
                attn_map_2 = torch.chunk(attn_map_2, batch_size)[idx]
                # .squeeze() # chunk의 첫번째가 bbox 마스크, 두번째가 tss 마스크
                upsacled_attn_map_1 = upscale(attn_map_1, image_size)
                upsacled_attn_map_2 = upscale(attn_map_2, image_size)
                net_attn_maps[name + "bbox"].append(upsacled_attn_map_1)
                net_attn_maps[name + "tss"].append(upsacled_attn_map_2)
    else:
        for name, attn_map_list in target_attn_map_dict.items():
            if attn_map_list is None:
                continue

            for attn_map in attn_map_list:
                attn_map = attn_map.cpu() if detach else attn_map
                attn_map = torch.chunk(attn_map, batch_size)[
                    idx
                ].squeeze()  # chunk의 첫번째가 bbox 마스크, 두번째가 tss 마스크
                upsacled_attn_map = upscale(attn_map, image_size)
                net_attn_maps[name].append(upsacled_attn_map)

    net_attn_maps = {
        key: torch.mean(torch.stack(value, dim=0), dim=0)
        for key, value in net_attn_maps.items()
    }
    return net_attn_maps


def attnmaps2images(net_attn_maps, w=512, h=512):

    # total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        # total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (
            (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        )
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        # print("norm: ", normalized_attn_map.shape)
        # resize to 512, 512
        normalized_attn_map = cv2.resize(
            normalized_attn_map, (w, h), interpolation=cv2.INTER_LANCZOS4
        )
        image = Image.fromarray(normalized_attn_map)

        # image = fix_save_attn_map(attn_map)
        images.append(image)

    # print(total_attn_scores)
    return images


def attnmaps2rgbimages(
    attn_maps: torch.Tensor,
    source_image: np.ndarray,
    h: int = 512,
    w: int = 512,
):

    source_image = cv2.resize(source_image, (w, h))
    images = []

    for attn_map in attn_maps:

        attn_map = attn_map.cpu().numpy()
        # total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (
            np.max(attn_map) - np.min(attn_map) + 1e-8
        )
        # normalized_attn_map = 1.0 - normalized_attn_map

        heatmap = cv2.applyColorMap(
            np.uint8(255 * normalized_attn_map), cv2.COLORMAP_JET
        )
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LANCZOS4)

        alpha = 0.85
        blended_image = cv2.addWeighted(source_image, 1 - alpha, heatmap, alpha, 0)
        blended_image = Image.fromarray(blended_image)
        images.append(blended_image)

    return images


def save_attn_heat_maps_with_prompt(
    attn_maps: torch.Tensor,
    source_image: np.ndarray,
    tokenizer,
    prompt,
    dir_name: str,
    h: int = 512,
    w: int = 512,
):

    source_image = cv2.resize(source_image, (w, h))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    normalized_source_image = np.float32(
        (source_image - np.min(source_image))
        / (np.max(source_image) - np.min(source_image) + 1e-8)
    )

    tokens = prompt2tokens(tokenizer, prompt)
    total_attn_scores = 0
    for i, (token, attn_map) in enumerate(zip(tokens, attn_maps)):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()
        attn_map_w, attn_map_h = attn_map.shape
        attn_map_total = attn_map_w * attn_map_h
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score
        token = token.replace("</w>", "")

        normalized_attn_map = (attn_map - np.min(attn_map)) / (
            np.max(attn_map) - np.min(attn_map) + 1e-8
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255 * normalized_attn_map), cv2.COLORMAP_JET
        )
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LANCZOS4)

        heatmap = np.float32(heatmap) / 255

        blended = heatmap + normalized_source_image

        blended = blended / np.max(blended)

        vis = np.uint8(255 * blended)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        save_attn_map(
            vis,
            f"{token}:{attn_map_score:.2f}",
            f"{dir_name}/{i}_<{token}>:{int(attn_map_score*100)}.png",
        )


def prompt2tokens(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    tokens = []
    for text_input_id in text_input_ids[0]:
        token = tokenizer.decoder[text_input_id.item()]
        tokens.append(token)
    return tokens


def save_net_attn_maps(net_attn_maps, dir_name, tokenizer, prompt):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    tokens = prompt2tokens(tokenizer, prompt)
    total_attn_scores = 0
    for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
        attn_map_score = torch.sum(attn_map)
        attn_map = attn_map.cpu().numpy()
        h, w = attn_map.shape
        attn_map_total = h * w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score
        token = token.replace("</w>", "")
        save_attn_map(
            attn_map,
            f"{token}:{attn_map_score:.2f}",
            f"{dir_name}/{i}_<{token}>:{int(attn_map_score*100)}.png",
        )
    print(f"total_attn_scores: {total_attn_scores}")


def resize_net_attn_map(net_attn_maps, target_size):
    net_attn_maps = F.interpolate(
        net_attn_maps.to(dtype=torch.float32).unsqueeze(0),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze()  # (77,64,64)
    return net_attn_maps


def save_attn_map(attn_map, title, save_path):
    normalized_attn_map = (
        (attn_map - np.min(attn_map))
        / (np.max(attn_map) - np.min(attn_map) + 1e-8)
        * 255
    )
    normalized_attn_map = normalized_attn_map.astype(np.uint8)
    image = Image.fromarray(normalized_attn_map)
    image.save(save_path, format="PNG", compression=0)


def return_net_attn_map(net_attn_maps, tokenizer, prompt):
    tokens = prompt2tokens(tokenizer, prompt)
    total_attn_scores = 0
    images = []
    for i, (token, attn_map) in enumerate(zip(tokens, net_attn_maps)):
        attn_map_score = torch.sum(attn_map)
        h, w = attn_map.shape
        attn_map_total = h * w
        attn_map_score = attn_map_score / attn_map_total
        total_attn_scores += attn_map_score

        attn_map = attn_map.cpu().numpy()
        normalized_attn_map = (
            (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        )
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        image = Image.fromarray(normalized_attn_map)

        token = token.replace("</w>", "")
        images.append((image, f"{i}_<{token}>"))
    print(f"total_attn_scores: {total_attn_scores}")
    return images


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
