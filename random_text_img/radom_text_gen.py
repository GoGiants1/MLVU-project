#pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오  
from tools import draw_centers_with_text
from tools import mask_size 
from tools import find_white_centers 
from PIL import Image 
import numpy as np 
from hi_sam_test.hisam import run_text_detection 


def random_text_img_gen(image_array,sample_text):
    
    #input : numpy array
    #sample_text : 그냥 " " 에 감싼 텍스트, 가장 밑 scene_text_image 에 해당 sample_text 만 나옴 

    sample_text=sample_text
    mask=run_text_detection(image_array)
    masks=mask.squeeze(1)
    new_mask=np.zeros_like(masks[0,:,:])

    for mask in masks:
        new_mask+=mask

    new_mask=Image.fromarray(new_mask)
    new_mask2=new_mask

    font_size_list=mask_size(new_mask2)
    centers=find_white_centers(new_mask2)
    font_size_list=[factor[1] for factor in font_size_list]

    text_contents=[sample_text]*len(font_size_list)
    scene_text_image=draw_centers_with_text(new_mask2,centers,text_contents,font_size_list)

    return scene_text_image # pillow image
