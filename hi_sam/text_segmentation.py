import gc
import os
import random
from dataclasses import dataclass, field

import numpy as np
import PIL
import torch
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from tqdm import tqdm

from hi_sam.models.auto_mask_generator import AutoMaskGenerator
from hi_sam.models.build import model_registry
from hi_sam.models.predictor import SamPredictor
from hi_sam.utils.patchify import patchify_sliding, unpatchify_sliding


@dataclass
class HisamArgs:
    checkpoint: str = field(
        metadata={"help": "The path to the SAM checkpoint to use for mask generation."}
    )
    model_type: str = field(
        default="vit_l",
        metadata={"help": "The type of model to load, in ['vit_h', 'vit_l', 'vit_b']"},
    )
    device: str = field(
        default="cuda", metadata={"help": "The device to run generation on."}
    )
    hier_det: bool = field(
        default=False, metadata={"help": "If False, only text stroke segmentation."}
    )
    input_size: tuple[int, int] = field(
        default_factory=lambda: [1024, 1024], metadata={"help": "The input image size."}
    )
    patch_mode: bool = field(default=False, metadata={"help": "self-prompting"})
    attn_layers: int = field(
        default=1,
        metadata={
            "help": "The number of image to token cross attention layers in model_aligner"
        },
    )
    prompt_len: int = field(default=12, metadata={"help": "The number of prompt token"})
    zero_shot: bool = field(
        default=False, metadata={"help": "If True, use zero-shot setting."}
    )
    vis: bool = field(default=False, metadata={"help": "If True, save visualization."})
    dataset: str = field(
        default="totaltext", metadata={"help": "Trained dataset for text detection"}
    )
    layout_thresh: float = field(default=0.5)
    seed: int = field(default=42)


def make_text_segmentation_args(
    model_type: str,
    checkpoint_path: str,
    input_size: tuple[int, int] = (1024, 1024),
    dataset: str = "totaltext",
    zero_shot: bool = False,
    hier_det: bool = True,
):
    """
    Create a HisamArgs object for text segmentation.

    args:
        `checkpoint_path`: The local path to the checkpoint to use for text segmentation or filename in hf_hub.
        https://huggingface.co/GoGiants1/Hi-SAM/tree/main
    """
    ckpt_path = (
        checkpoint_path
        if os.path.isfile(checkpoint_path)
        else hf_hub_download(
            repo_id="GoGiants1/Hi-SAM",
            repo_type="model",
            filename=checkpoint_path,
        )
    )

    args = HisamArgs(
        model_type=model_type,
        checkpoint=ckpt_path,
        dataset=dataset,
        input_size=input_size,
        hier_det=hier_det,
        zero_shot=zero_shot,
        layout_thresh=0.5,
        patch_mode=True,
    )
    return args


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


def load_auto_mask_generator(args: HisamArgs):
    set_random_seed(args.seed)
    empty_cuda_cache()
    model = model_registry[args.model_type](args)
    model.eval()
    model.to(args.device)
    amg = AutoMaskGenerator(model)
    return amg


def load_sam_predictor(args: HisamArgs):
    set_random_seed(args.seed)
    empty_cuda_cache()
    model = model_registry[args.model_type](args)
    model.eval()
    model.to(args.device)
    sam = SamPredictor(model)
    return sam


def model_to_device(amg: AutoMaskGenerator | SamPredictor, device: str = "cpu"):
    amg.model.to(device)
    empty_cuda_cache()
    return amg


def unload_model(amg: AutoMaskGenerator):
    if amg.model is not None:
        amg.model.cpu()
    empty_cuda_cache()
    return amg


def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = (
            color
            if color is not None
            else np.array([30 / 255, 144 / 255, 255 / 255, 0.5])
        )
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks(masks, filename, image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    for i, mask in enumerate(masks):
        mask = mask[0].astype(np.uint8)
        # contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for cont in contours:
        #     epsilon = 0.002 * cv2.arcLength(cont, True)
        #     approx = cv2.approxPolyDP(cont, epsilon, True)
        #     pts = approx.reshape((-1, 2))
        #     if pts.shape[0] < 4:
        #         continue
        #     pts = pts.astype(np.int32)
        #     mask = cv2.fillPoly(np.zeros(mask.shape), [pts], 1)
        show_mask(mask, plt.gca(), random_color=True)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_text_detection(
    amg: AutoMaskGenerator,
    image: PIL.Image.Image | np.ndarray,
    from_low_res: bool = False,
    fg_points_num: int = 1500,
    batch_points_num: int = 100,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    zero_shot: bool = False,
    dataset: str = "totaltext",
):
    """
    Run text detection on an image using the given AutoMaskGenerator.

    Args:
        `amg`: The AutoMaskGenerator to use for text detection.
        `image`: The image  to run text detection on.
        `fg_points_num`: The number of foreground points to use.
        `batch_points_num`: The number of points to process in each batch.
        `score_thresh`: The score threshold to use for text detection.
        `nms_thresh`: The NMS threshold to use for text detection.
        `zero_shot`: If True, use zero-shot setting.
        `dataset`: The dataset to use for text detection.

    Returns:
        `masks`: The text masks. (number of detected words or lines, height, width)
        `scores`: The scores for each mask. (number of detected words or lines,)
    """

    # resize to 1024x1024
    # if image.size[0] > 1024 or image.size[1] > 1024:
    #     image = image.resize((1024, 1024), Image.Resampling.LANCZOS)

    image_arr = np.array(image) if isinstance(image, PIL.Image.Image) else image
    amg.set_image(image_arr)

    masks, scores = amg.predict_text_detection(
        from_low_res=from_low_res,
        fg_points_num=fg_points_num,
        batch_points_num=min(batch_points_num, 100),
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        zero_shot=zero_shot,
        dataset=dataset,
    )

    return masks, scores


def run_text_stroke_segmentation(
    sam_detector: SamPredictor,
    image: PIL.Image.Image | np.ndarray,
    patch_mode: bool = False,
):
    mask = None
    image_arr = np.array(image) if isinstance(image, PIL.Image.Image) else image
    if patch_mode:
        ori_size = image_arr.shape[:2]
        patch_list, h_slice_list, w_slice_list = patchify_sliding(
            image_arr, 512, 384
        )  # sliding window config
        mask_512 = []
        for patch in tqdm(patch_list):
            sam_detector.set_image(patch)
            m, hr_m, score, hr_score = sam_detector.predict(
                multimask_output=False, return_logits=True
            )
            assert hr_m.shape[0] == 1  # high-res mask
            mask_512.append(hr_m[0])
        mask_512 = unpatchify_sliding(mask_512, h_slice_list, w_slice_list, ori_size)
        assert mask_512.shape[-2:] == ori_size
        mask = mask_512
        mask = mask > sam_detector.model.mask_threshold
        print("Mask shape: ", mask.shape)
    else:
        mask, hr_mask, score, hr_score = sam_detector.predict(multimask_output=False)
        print("Mask shape: ", mask.shape)

    return mask
