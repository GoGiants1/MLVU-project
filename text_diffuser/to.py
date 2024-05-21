import torch
import numpy as np 

a=torch.rand(4,3,100,100)
sample_images2=torch.permute(a,(0,2,3,1))
sample_images2=np.array(sample_images2)
print(sample_images2.shape)
print(type(sample_images2))
# clip score 계산하기 