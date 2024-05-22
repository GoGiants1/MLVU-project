import cv2
import torch
from PIL import Image
from t_diffusers.unet_2d_condition import UNet2DConditionModel

from diffusers import AutoencoderKL, DDPMScheduler

from pipeline_text_diffuser_sd15 import StableDiffusionPipeline

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
pipe.set_ip_adapter_scale(1.0)

text_mask_image = cv2.imread("assets/examples/text-inpainting/case3_mask.jpg") ## 수용님이 만드신 마스크 이미지 혹은 넘파이 배열이 들어가야합니다.

input_image = Image.open("assets/examples/text-inpainting/case3.jpg").convert("RGB").resize((512, 512))

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
    generator=generator,
    save_output = True,
    num_images_per_prompt = 4,
).images[0]