import easyocr
import os
import argparse
from PIL import Image
import numpy as np
import Levenshtein as lev
import re
class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __repr__(self) -> str:
        return str(self.avg)

class OCR_EM_Counter(object):
    '''Computes and stores the OCR Exactly Match Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_acc_em = {}
        self.ocr_acc_em_rate = 0

    def add_text(self, text):
        if text not in self.ocr_acc_em:
            self.ocr_acc_em[text] = AverageMeter()
        
    def update(self, text, ocr_result):
        ocr_texts = [item[1] for item in ocr_result]
        self.ocr_acc_em[text].update(text in ocr_texts)
        self.ocr_acc_em_rate = sum([value.sum for value in self.ocr_acc_em.values()]) / sum([value.count for value in self.ocr_acc_em.values()])
    
    def __repr__(self) -> str:
        ocr_str = ",".join([f"{key}:{repr(value)}" for key, value in self.ocr_acc_em.items()])
        return str(self.ocr_acc_em_rate)
        # return f"OCR EM Accuracy is {self.ocr_acc_em_rate}."
    
    
class OCR_EM_without_capitalization_Counter(object):
    '''Computes and stores the OCR Exactly Match Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_acc_em = {}
        self.ocr_acc_em_rate = 0

    def add_text(self, text):
        if text not in self.ocr_acc_em:
            self.ocr_acc_em[text] = AverageMeter()
        
    def update(self, text, ocr_result):
        ocr_texts = [item[1].lower() for item in ocr_result]
        self.ocr_acc_em[text].update(text.lower() in ocr_texts)
        self.ocr_acc_em_rate = sum([value.sum for value in self.ocr_acc_em.values()]) / sum([value.count for value in self.ocr_acc_em.values()])
    
    def __repr__(self) -> str:
        ocr_str = ",".join([f"{key}:{repr(value)}" for key, value in self.ocr_acc_em.items()])
        return str(self.ocr_acc_em_rate)

class OCR_Levenshtein_Distance(object):
    '''Computes and stores the OCR Levenshtein Distance Accuracy.'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_lev = {}
        self.ocr_lev_avg = 0

    def add_text(self, text):
        if text not in self.ocr_lev:
            self.ocr_lev[text] = AverageMeter()
    
    def update(self, text, ocr_result):
        ocr_texts = [item[1] for item in ocr_result]
        lev_distance = [lev.distance(text, ocr_text) for ocr_text in ocr_texts]
        if lev_distance:
            self.ocr_lev[text].update(min(lev_distance))
            self.ocr_lev_avg = sum([value.sum for value in self.ocr_lev.values()]) / sum([value.count for value in self.ocr_lev.values()])

    def __repr__(self) -> str:
        return str(self.ocr_lev_avg)
    
#if __name__ == "__main__":

def ocr_eval(image_array,txt):
    
    #image_array: text를 통해서 생성된이미지(scene text가 불안정하게 나온 이미지)
    #image_array: numpy array, item: 이미지에 나와야할 text 내용
    
    match = re.search(r"'(.*?)'", txt)    
    item = match.group(1)
    text=item
    
    print(f"Scene text is : {text}")
    reader = easyocr.Reader(['en'])
    
    sum=0
    sum2=0
    sum3=0
    
    for i in range(image_array.shape[0]):
        
        ocr_em_counter = OCR_EM_Counter()
        ocr_em_wc_counter = OCR_EM_without_capitalization_Counter()
        ocr_lev = OCR_Levenshtein_Distance()
        
        ocr_em_counter.add_text(text)
        ocr_em_wc_counter.add_text(text)
        ocr_lev.add_text(text)
        
        ocr_result = reader.readtext(image_array[i])
        print(f"Read text is : {ocr_result}")
    
        ocr_em_counter.update(text, ocr_result)
        ocr_em_wc_counter.update(text, ocr_result)
        ocr_lev.update(text, ocr_result)
        
    
        sum+=float(ocr_em_counter.ocr_acc_em_rate)
        sum2+=float(ocr_em_wc_counter.ocr_acc_em_rate)
        sum3+=float(ocr_lev.ocr_lev_avg)
        
        
    
    return sum/image_array.shape[0],sum2/image_array.shape[0],sum3/image_array.shape[0]

if __name__ == "__main__":
    import torch
    img=torch.rand(224,224,3)
    img=Image.open("/root/clip_fid/MLVU-project-main/text_diffuser/wendy2.png")
    img=np.array(img)
    #print(img.shape)
    img=torch.from_numpy(img)
    img=img.unsqueeze(0)
    img=np.array(img)
    
    a,b,c=ocr_eval(img,"dog and man 'wendy' walk")
    print(a,b)
    
    
    # a, 대문자 소문자 까지 따짐
    # b, 대문자 소문자 무관 
    # c, leven distance
