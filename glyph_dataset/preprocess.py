import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
from tqdm import tqdm
def draw_centers_with_text(sample_texts,files):

    # storke: (512,512)차원에 각 값은 0-255 1차원. 검정색 부분이 글자
    # masks: 검정색 배경 하얀색 mask
    # center_ls: mask들의 중심좌표 

    """ 1. 검은색 배경에 masks들을 하나하나 그려준다 """
    images = np.zeros((len(sample_texts)*5,128,128), dtype=np.uint8)
    for font_file in files:
        os.makedirs(f"experiment/{font_file}", exist_ok=True)
    """ 2. coordinates에서 한 좌표씩 뽑고, 그 좌표와 가장 가까운 scene text 부분만 sample_text로 대체가 된다 """
    for i in tqdm(range(len(sample_texts))):
        for j in range(5):
            # 해당 mask의 폰트 사이즈를 찾아서, sample_text를 그 폰트 사이즈에 맞게 그림
            image = Image.fromarray(images[i*5+j])
            font_size = random.randint(20,90)
            vertical_prob = random.uniform(0,1)
            font_num = random.randint(0, len(files)-1)
            font = ImageFont.truetype(f"font/{files[font_num]}", font_size)
            text_image = Image.new('L', (128, 128), 0)
            text_draw = ImageDraw.Draw(text_image)
            # 해당 폰트 사이즈로 sample_text를 그려서, 그 길이와 높이를 구함
            if(vertical_prob<0.8):
                angle = random.randint(-90,90)
                _, _, text_w, text_h = font.getbbox(sample_texts[i])
               
        
        # 보통 폰트 사이즈가 너무 크기에, 폰트 사이즈를 줄여줌 
                if text_w>128:
                    font_size *= 96/text_w
                    font = ImageFont.truetype(f"font/{font_file}", font_size)
                    _, _, text_w, text_h = font.getbbox(sample_texts[i])
       
        # 512, 512 이미지 중앙에 텍스트를 그림
                
                text_draw.text(((128-text_w)/2, (128-text_h)/2), sample_texts[i], font=font, fill=255)
            else:
                angle = random.randint(-30,30)
                text_h =[]
                for chrac in sample_texts[i]:
                    _, _, char_w, char_h = font.getbbox(chrac)
                    text_h.append(char_h)
                
                if sum(text_h)>128:
                    font_size *= 96/sum(text_h)
                    font = ImageFont.truetype(f"font/{font_file}", font_size)
                    text_h =[]
                    for chrac in sample_texts[i]:
                        _, _, char_w, char_h = font.getbbox(chrac)
                        text_h.append(char_h)
                height = 0   
                for k in range(len(text_h)):
                    text_draw.text(((128-char_w)/2, (128-sum(text_h))/2+height), sample_texts[i][k], font=font, fill=255)
                    height += text_h[k]
                
                
        # 둘 다 회전. 이미지 중앙을 기준으로 회전한 것이라, 제자리에서 잘 회전합니다.
            text_rotate = text_image.rotate(angle, expand=0)
        
        # 글씨를 이제 paste하는데 center_ls[the_index]가 중심이 되도록 paste한다.
        
        # 얼마나 움직여야하는지 아래와 같이 계산. 
        
        # 글씨 붙여넣기
            image.paste(text_rotate, (0,0))
            if len(sample_texts[i])>5:
                if isinstance(image, Image.Image):
                    image.save(f"experiment/{files[font_num]}/{sample_texts[i]}.jpg")

            images[i*5+j] = np.array(image)
    
    """ 3. 최종 리턴할 그림에서 stroke가 0인 부분은 검은색으로 """
    
    return images

