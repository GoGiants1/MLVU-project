#pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오
import os
import sys

import numpy as np
from mask_only_tools import draw_centers_with_text, find_white_centers, mask_size, sorting1
from PIL import Image


sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from hi_sam.text_segmentation import (
    load_auto_mask_generator,
    load_sam_predictor,
    run_text_detection,
    run_text_stroke_segmentation,
)


def gen_mask_only(image, sample_text, choice_list, arg_textseg, arg_maskgen):

    #input : numpy array
    #sample_text : 그냥 " " 에 감싼 텍스트, 가장 밑 scene_text_image 에 해당 sample_text 만 나옴

    """""

    choice_list 는 scene text 가 가장 위 나 왼쪽에 있는것을 choice_list의 첫번째 요소로 간주한다. 이때 choice_list의 첫요소가 1이면 
    scene text 중 가장 왼쪽, 가장 위인것만 output에서 sample text가 나타난다.
    나머지 choice list에서 0인것들은 text stroke 된것이 나타난다

    """""
    amg = load_auto_mask_generator(arg_maskgen)
    sam = load_sam_predictor(args=arg_textseg)

    mask, _ = run_text_detection(amg=amg, image=image)
    masked_text_stroke = run_text_stroke_segmentation(sam_detector=sam, image=image, patch_mode=True)

    #tss = Image.fromarray(masked_text_stroke)
    #tss.save("tss.png")
    stroke_mask = np.where(masked_text_stroke == True, 255, 0)

    # mask3_image = Image.fromarray(mask3, "L")
    # save mask3
    # mask3_image.save("mask3.png")

    #검정색 글씨 하얀배경
    masks=mask.squeeze(1)
    #검정색 배경 하얀 mask
    new_mask=np.sum(masks,axis=0)
    new_mask = 255-new_mask
    Image.fromarray(masks[0], "L").save("assets/examples/text-inpainting/new_mask.png")
    Image.fromarray(stroke_mask, "L").save("assets/examples/text-inpainting/stroke_mask.png")
    exit()
    


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
