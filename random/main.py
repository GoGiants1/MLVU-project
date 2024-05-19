#pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오  
from tools import draw_centers_with_text, mask_size, find_white_centers,sorting1
from PIL import Image 
import numpy as np 
from hi_sam_test.hisam import run_text_detection 
from hi_sam.run_hisam import text_stroke

def random_text_img_gen(image_array,sample_text,choice_list):
    
    #input : numpy array
    #sample_text : 그냥 " " 에 감싼 텍스트, 가장 밑 scene_text_image 에 해당 sample_text 만 나옴 

    """""

    choice_list 는 scene text 가 가장 위 나 왼쪽에 있는것을 choice_list의 첫번째요소로간주한다. 이때 choice_list의 첫요소가 1이면 
    scene text 중 가장 왼쪽, 가장 위인것만 output에서 sample text가 나타난다.
    나머지 choice list에서 0인것들은 text stroke 된것이 나타난다

    """""

    sample_text=sample_text
    mask=run_text_detection(image_array)
    mask2=text_stroke(image_array)

   
    mask2=mask2.squeeze(0)
    mask3=np.zeros_like(mask2)
    for w in range(mask2.shape[0]):
        for h in range(mask2.shape[1]):
            if mask2[w,h]==True:
                mask3[w,h]=0.0
            else:
                mask3[w,h]=255.0
    
    stroke_mask=mask3
    #mask3=Image.fromarray(mask)
    #검정색 글씨 하얀배경 
    #stroke_mask=stroke(image_array)
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
    scene_text_image=draw_centers_with_text(image_array,new_mask2,centers,text_contents,font_size_list,choice_list,stroke_mask)

    return scene_text_image 


if __name__ == "__main__":

    img=Image.open("scene2.png")    
    img=np.array(img)
    print(img.shape)
    
    sample_text="bear"
    coordinates=[[110,200]]

    out=random_text_img_gen(img,sample_text=sample_text,choice_list=coordinates)
    out.save(f"scene2_out.png")