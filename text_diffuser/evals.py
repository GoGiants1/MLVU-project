import torch
from datasets import load_dataset
import numpy as np

import sys
sys.path.append('/root/MLVU-project')
from eval.clipscore.clipscore import cal_clip_score
from eval.ocr import ocr_eval
import  pytorch_fid_wrapper as pfw
import requests 


def get_one_image_and_caption(number):
    dataset = load_dataset("GoGiants1/TMDBEval500")
    img=dataset["train"][number]["image"]
    link ="https://huggingface.co/datasets/GoGiants1/TMDBEval500/resolve/main/TMDBEval500/TMDBEval500.txt"
    caption_link=requests.get(link)
    captions=caption_link.text
    lines=captions.splitlines()
    cap_list=[]
    for index,line in enumerate(lines):
        if index==number:
            cap_list.append(line)
            break
    
    return img,cap_list[0]

def cal_fid_score(input,target):
    

    input=np.array(input)
    input=torch.from_numpy(input)
    input=input.permute(2,0,1)
    input=input.unsqueeze(0)

    target_image=np.array(target)
    target=torch.from_numpy(target_image)
    target=target.permute(2,0,1)
    target=target.unsqueeze(0)

    batch=input.shape[0]
    pfw.set_config(batch_size=batch, dims=2048, device="cuda")

    real_images=input
    fake_images=target

    real_m, real_s = pfw.get_stats(real_images)
    val_fid = pfw.fid(fake_images, real_m=real_m, real_s=real_s) 

    return val_fid



def print_clip_score_ocr_score(input,caption):
    input=np.array(input)
    input=torch.from_numpy(input)
    input=input.unsqueeze(0)
    output=np.array(input)
    clip_score=cal_clip_score(output,caption)
    
    print("CLIP SCORE")
    print(clip_score)

    ocr_score=ocr_eval(output,caption)
    print("OCR SCORE")
    print("ocr_em_counter, 대문자 소문자 구별")
    print(ocr_score[0])
    print("ocr_em_counter, 대문자 소문자 구별안함")
    print(ocr_score[1])
    print("ocr lev")
    print(ocr_score[2])



     
    
    