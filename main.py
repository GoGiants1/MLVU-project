import torch
import sys
import argparse
import numpy as np
from utils import load_img_to_array,save_array_to_img
from hi_sam.box_mask import making_mask


def setup_args(parser):
    
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--output_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        help="remove or replace"
    )
    
  


if __name__ == "__main__":
    
    from PIL import Image
    #from stable_diffusion_inpaint import replace_img_with_sd
    
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)
    mask=making_mask(img)
    
    #위 img, mask는 둘다 np array
    
    if args.mode.lower()=="replace":
        from stable_diffusion_inpaint import replace_img_with_sd
    
        img=Image.fromarray(img)

        img=np.array(img)
        mask=np.array(mask)
    
        mask2=np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for i2 in range(mask.shape[1]):
                if mask[i,i2]==True:
                    mask2[i,i2]=0.0
                else:
                    mask2[i,i2]=255.0
                                    
        result=replace_img_with_sd(img,mask2,text_prompt="green")
        result=result.astype(np.uint8)
        result=Image.fromarray(result)
        result.save(args.output_img)
    else:
        from lama_inpaint import inpaint_img_with_lama

        mask=np.array(mask)
        mask2=np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for i2 in range(mask.shape[1]):
                if mask[i,i2]==True:
                    mask2[i,i2]=255.0
                else:
                    mask2[i,i2]=0.0
                    
        img_inpainted = inpaint_img_with_lama(
            img, mask2, "./lama/configs/prediction/default.yaml", "pretrained_models/big-lama", device=device)
        save_array_to_img(img_inpainted,args.output_img)
    
    
        
        
        
        

