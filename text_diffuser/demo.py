import cv2
import torch
from PIL import Image
from t_diffusers.unet_2d_condition import UNet2DConditionModel
from generate_mask_only import gen_mask_only
from t_diffusers.unet_2d_condition import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import DDPMScheduler
from pipeline_text_diffuser_sd15 import StableDiffusionPipeline
from hi_sam.text_segmentation import make_text_segmentation_args
import numpy as np

td_ckpt = "textdiffuser-ckpt/diffusion_backbone_1.5"


unet = UNet2DConditionModel.from_pretrained(
    td_ckpt,
    subfolder="unet",
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float32,
)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder=[
        "models",
    ],
    weight_name=[
        "ip-adapter_sd15.safetensors",
    ],
)
pipe.set_ip_adapter_scale(0.9)

input_image = Image.open("assets/examples/text-inpainting/case3.jpg").convert("RGB").resize((512, 512))

sample_text="jaehak"
# for original_input.jpeg. 110, 500에서 가장 가까운 mask의 글자를 바꾼다.
coordinates=[[110, 500]] 

arg_textseg = make_text_segmentation_args(
    model_type='vit_h',
    checkpoint_path='sam_tss_h_textseg.pth',
    input_size=(512, 512),
    hier_det=False,
)

arg_maskgen = make_text_segmentation_args(
    model_type='vit_h',
    checkpoint_path='word_detection_totaltext.pth',
    input_size=(512, 512),
    hier_det=True,
)

out = gen_mask_only(input_image, sample_text=sample_text, choice_list=coordinates, arg_textseg=arg_textseg, arg_maskgen=arg_maskgen)
out.save(f"./assets/examples/text-inpainting/mask_3_out.png") # for debugging
text_mask_image= cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator(device="cuda").manual_seed(42)
pipe.to("cuda")
output = pipe(
    prompt="a red board saying 'Sessions'",
    input_image=input_image,
    text_mask_image=text_mask_image,
    ip_adapter_image = input_image,
    width=512,
    height=512,
    guidance_scale=9.5,

    save_output = True,
    num_images_per_prompt = 4,
).images[0]