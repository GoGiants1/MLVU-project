from dataclasses import dataclass, field
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from tqdm import tqdm
from PIL import Image 
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


def run_text_detection(array):

    seed = 42
    torch.manual_seed(seed)

    model = model_registry[args.model_type](args)
    model.eval()
    model.to(args.device)

    amg = AutoMaskGenerator(model)
    image_arr = array

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
    
    return masks


if __name__ == "__main__":
    run_text_detection()
