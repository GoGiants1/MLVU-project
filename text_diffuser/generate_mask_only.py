#pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오
import os
import sys

import numpy as np
from mask_only_tools import draw_centers_with_text, take_info


sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

from hi_sam.text_segmentation import (
    load_auto_mask_generator,
    load_sam_predictor,
    run_text_detection,
    run_text_stroke_segmentation,
)


def gen_mask_only(image, sample_text, coordinates, arg_textseg, arg_maskgen):

    #input : numpy array
    #sample_text : 그냥 " " 에 감싼 텍스트, 가장 밑 scene_text_image 에 해당 sample_text 만 나옴
    #coordinates: user가 선택한 editing 좌표
    #arg_textseg: text segmentation model argument
    #arg_maskgen: mask generation model argument

    """""

    choice_list 는 scene text 가 가장 위왼쪽에 있는것을 choice_list의 첫번째 요소로 간주한다. 이때 choice_list의 첫요소가 1이면 
    scene text 중 가장 왼쪽, 가장 위인것만 output에서 sample text가 나타난다.
    나머지 choice list에서 0인것들은 text stroke 된것이 나타난다

    """""
    amg = load_auto_mask_generator(arg_maskgen)
    sam = load_sam_predictor(args=arg_textseg)

    masks, _ = run_text_detection(amg=amg, image=image)
    masked_text_stroke = run_text_stroke_segmentation(sam_detector=sam, image=image, patch_mode=True)

    """ 1. 흰색 배경에 검은색 text stroke 생성 """
    stroke_mask = np.array(np.where(masked_text_stroke == True, 0, 255), dtype=np.uint8)

    """ 2. 검은색 배경에 흰색 마스크들을 그룹으로 가져온다. 예를 들어 마스크가 34개 있으면 masks의 최종 차원은 (34, 512, 512) """
    masks = np.array(masks.squeeze(1),dtype=np.uint8)
    masks = masks*255

    """ 3. 각 mask의 중심좌표, 폰트 사이즈, 각도 정보 추출 by 가장 큰 contour, 해당 contour에서 가장 작은 직사각형 생성 """
    center_ls, font_size_ls, angle_ls = take_info(masks)
    # font_size_list: (마스크의 너비, 마스크의 높이)

    """ 4. 유저 마음대로 x y 좌표를 정하고(coordinates) 그 좌표에 가장 가까운 부분의 mask 부분만 bear로 바꾸기. 나머지부분은 text stroke 유지 """
    scene_text_image = draw_centers_with_text(masks, center_ls, angle_ls, sample_text, font_size_ls, coordinates, stroke_mask)

    return scene_text_image
