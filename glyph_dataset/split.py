import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os


image = Image.open('./glyph1.jpg')
image_np = np.array(image)
split_list = np.split(image_np, 8, axis=0)
split_image_list = [np.split(split, 8, axis=1) for split in split_list]
split_image_list =  [item for sublist in split_image_list for item in sublist]
for i in range(len(split_image_list)):
    image_split = Image.fromarray(split_image_list[i])
    image_split.save(f"./split/{i}.jpg")