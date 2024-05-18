import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def find_white_centers(image):
    
    image = image.convert('L')  
    np_image = np.array(image)
    
    _, binary_image = cv2.threshold(np_image, 254, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    centers = []
    
    for i in range(1, num_labels):
        center_x = int(centroids[i][0])
        center_y = int(centroids[i][1])
        centers.append((center_x, center_y))
    
    return centers

def draw_centers_with_text(mask,centers,text_contents,font_size_list):
    
    mask=np.array(mask)
    image0 = np.ones_like(mask)
    image=Image.fromarray(image0)
    image=image.convert("L")
    image=np.array(image)
    for w in range(image.shape[0]):
        for h in range(image.shape[1]):
            if mask[w,h]==1.0:
                image[w,h]=128.0
            else:
                continue

    image=Image.fromarray(image)
    image=image.convert("L")
    draw = ImageDraw.Draw(image)
    
    for idx, (x, y) in enumerate(centers):

        font = ImageFont.truetype("/root/mlvu/MLVU-project/random_text_img/Arial.ttf", font_size_list[idx]) 
        text = text_contents[idx]
        _, _, text_width, text_height = font.getbbox(text)  

        text_x = x - text_width // 2
        text_y = y - text_height // 2
        
        draw.text((text_x, text_y), text, font=font, fill="black")

    return image #pillow 

def mask_size(image):

    image = image.convert('L')
    np_image = np.array(image)
    
    _, binary_image = cv2.threshold(np_image, 254, 255, cv2.THRESH_BINARY)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    masks_size = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        masks_size.append((w, h))
    
    return masks_size


