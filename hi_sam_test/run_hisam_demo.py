import os
import random
from dataclasses import dataclass, field
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

from diffusers.utils import load_image
from hi_sam.models.auto_mask_generator import AutoMaskGenerator
from hi_sam.models.build import model_registry


def unclip(p, unclip_ratio=2.0):
    poly = Polygon(p)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(p, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def polygon2rbox(polygon, image_height, image_width):
    rect = cv2.minAreaRect(polygon)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1, 2)
    return pts


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


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


@dataclass
class Args:
    # input: List[str] = field(metadata={"help": "Path to the input image"})
    checkpoint: str = field(
        metadata={"help": "The path to the SAM checkpoint to use for mask generation."}
    )
    output: str = field(
        default="./demo",
        metadata={"help": "A file or directory to save output visualizations."},
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
    input_size: List[int] = field(
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
    vis: bool = field(default=True, metadata={"help": "If True, save visualization."})
    dataset: str = field(
        default="totaltext", metadata={"help": "Trained dataset for text detection"}
    )
    layout_thresh: float = field(default=0.5)


args = Args(
    model_type="vit_h",
    checkpoint="pretrained_checkpoint/hi_sam_h.pth",
    dataset="ctw1500",
    output="./demo_hi_sam_h_ctw1500_fg_points_1500_th_0.5",
    hier_det=True,
    vis=True,
    zero_shot=True,
    layout_thresh=0.5,
    input_size=[512, 512],
    patch_mode=True,
    attn_layers=1,
    prompt_len=12,
    device="cuda",
)


def run_text_detection():
    model = model_registry[args.model_type](args)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    model.to(args.device)
    print("Loaded model")
    amg = AutoMaskGenerator(model)

    if args.dataset == "totaltext":
        if args.zero_shot:
            fg_points_num = 50  # assemble text kernel # noqa: F841
            score_thresh = 0.3  # noqa: F841
            unclip_ratio = 1.5  # noqa: F841
        else:
            fg_points_num = 500
            score_thresh = 0.95
    elif args.dataset == "ctw1500":
        if args.zero_shot:
            fg_points_num = 100  # noqa: F841
            score_thresh = 0.6  # noqa: F841
        else:
            fg_points_num = 300  # noqa: F841
            score_thresh = 0.7  # noqa: F841
    else:
        raise ValueError

    # if os.path.isdir(args.input[0]):
    #     args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    # elif len(args.input) == 1:
    #     args.input = glob.glob(os.path.expanduser(args.input[0]))
    #     assert args.input, "The input path(s) was not found"
    for i in tqdm(range(500)):
        hf_dataset_base_url = "https://huggingface.co/datasets/GoGiants1/TMDBEval500/resolve/main/TMDBEval500/images/"
        url = hf_dataset_base_url + f"{i}.jpg"

        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

        assert os.path.isdir(args.output), args.output
        img_name = f"{i}.png"
        out_filename = os.path.join(args.output, img_name)

        image = load_image(url)
        image_arr = np.asarray(image)
        img_h, img_w = image_arr.shape[:2]

        amg.set_image(image_arr)
        masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=1500,
            batch_points_num=min(1500, 100),
            score_thresh=0.5,
            nms_thresh=0.5,
            zero_shot=args.zero_shot,
            dataset=args.dataset,
        )
        # print(masks.shape)
        # print(scores)

        if masks is not None:
            print("Inference done. Start plotting masks.")
            show_masks(masks, out_filename, image)
        else:
            print("no prediction")


if __name__ == "__main__":
    run_text_detection()
