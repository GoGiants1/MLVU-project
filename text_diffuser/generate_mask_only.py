#pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오  
from text_diffuser.mask_only_tools import draw_centers_with_text, mask_size, find_white_centers,sorting1
from PIL import Image 
import numpy as np 

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from hi_sam.text_segmentation import run_text_stroke_segmentation, load_sam_predictor, load_auto_mask_generator, run_text_detection


def gen_mask_only(image, sample_text, choice_list, arg_textseg, arg_maskgen):
    
    #input : numpy array
    #sample_text : 그냥 " " 에 감싼 텍스트, 가장 밑 scene_text_image 에 해당 sample_text 만 나옴 

    """""

    choice_list 는 scene text 가 가장 위 나 왼쪽에 있는것을 choice_list의 첫번째요소로간주한다. 이때 choice_list의 첫요소가 1이면 
    scene text 중 가장 왼쪽, 가장 위인것만 output에서 sample text가 나타난다.
    나머지 choice list에서 0인것들은 text stroke 된것이 나타난다

    """""
    amg = load_auto_mask_generator(arg_maskgen)
    sam = load_sam_predictor(args=arg_textseg)
    
    mask, _ = run_text_detection(amg=amg, image=image)
    masked_text_stroke = run_text_stroke_segmentation(sam_detector=sam, image=image, patch_mode=True)
   
    mask3=np.zeros_like(masked_text_stroke)
    for w in range(masked_text_stroke.shape[0]):
        for h in range(masked_text_stroke.shape[1]):
            if masked_text_stroke[w,h]==True:
                mask3[w,h]=0.0
            else:
                mask3[w,h]=255.0
    
    stroke_mask=mask3
    #검정색 글씨 하얀배경 
    masks=mask.squeeze(1)
    new_mask=np.zeros_like(masks[0,:,:])
    
    for mask in masks:
        new_mask+=mask

    new_mask=Image.fromarray(new_mask)
    new_mask2=new_mask
    
    font_size_list0=mask_size(new_mask2)
    centers=find_white_centers(new_mask2)
    centers=sorting1(centers)

    font_size_list=[factor[1] for factor in font_size_list0]
    # 유저 마음대로 x y 좌표를 정하고 그 좌표에 가장 가까운 부분의 scene text 부분만 bear로 바꾸기 , 나머지부분은 stroke 한걸로 하기 
    text_contents=sample_text
    scene_text_image=draw_centers_with_text(new_mask2,centers,text_contents,font_size_list,choice_list,stroke_mask)

    return scene_text_image