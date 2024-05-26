import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def draw_centers_with_text(masks,centers,text_contents,size_list,coordinates,stroke):

    #input_img :numpy array
    #storke: 0-255 1차원 , 검정색 부분이 글자 ,
    #mask: 검정색 배경 하얀색 mask
    #centers: mask들의 중심좌표 

    # text가 있는 영역들이 회색으로 칠해진다.

    image = np.ones((512,512),dtype=np.uint8)*255 
    image=Image.fromarray(image)
    image=image.convert("L")
    draw = ImageDraw.Draw(image)
    text_len = len(text_contents)
    # choice list내에 있는 좌표와 가장 가까운 scene text 부분만 bear 로 대체가 된다.
    for coord in coordinates:
        the_index=closest_index(coord,centers)
        stroke[np.where(masks[the_index]==255)] = 255
        size_tuple = size_list[the_index]
        w,h = size_tuple
        if w >h:
            size = min(h,w/text_len)
        else:
            size = min(w,h/text_len)
        font = ImageFont.truetype("assets/font/Arial.ttf", size)
        text = text_contents
        _, _, text_width, text_height = font.getbbox(text)

        text_x = centers[the_index][0] - text_width // 2
        text_y = centers[the_index][1] - text_height // 2
        draw.text((text_x, text_y), text, font=font, fill="black")

    image=fill_gray_with_text(image,stroke)

    return image

def closest_index(choice_list, centers):
    min_distance = float('inf')
    min_index = -1

    for i, center in enumerate(centers):
        distance = ((choice_list[0] - center[0])**2 + (choice_list[1] - center[1])**2)**0.5

        if distance < min_distance:
            min_distance = distance
            min_index = i

    return min_index


def sorting_coord(list):

    centers_with_sums = [(sum(tup), tup) for tup in list]
    sorted_centers_with_sums = sorted(centers_with_sums)
    sorted_centers = [tup for _, tup in sorted_centers_with_sums]

    return sorted_centers


def take_info(masks):
    #np_image는 검은 바탕에 흰색 박스들로 이루어진 이미지
    #_, binary_image = cv2.threshold(np_image, 254, 255, cv2.THRESH_BINARY)
    masks_size = []
    centers = []
    angle_list = [] 
    for i in range(masks.shape[0]):
        _, binary = cv2.threshold(masks[i], 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        if len(contours) > 1:
            for tmp in contours:
                if cv2.contourArea(tmp) > cv2.contourArea(contour):
                    contour = tmp
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        centers.append(center) #center is tuple (x, y) 
        masks_size.append((size[1],size[0])) #size is tuple (width, height)
        if size[1] > size[0]:
            angle = 90 -angle
        angle_list.append(angle)
    '''
    수용님 코드
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    masks_size = []
    centers = []
    for i in range(len(num_labels)):
        _,_, w, h, _ = stats[i]
        masks_size.append(min(w,h))
        center_x = int(centroids[i][0])
        center_y = int(centroids[i][1])
        centers.append((center_x, center_y))
    '''
    return (centers,masks_size,angle_list)



def fill_gray_with_text(mask_array, stroke_array):
    # PIL 이미지 배열과 numpy 배열의 크기가 같은지 확인
    assert mask_array.shape == stroke_array.shape, "The two arrays must have the same shape"

    result_array = pil_array.copy()
    mask_array=filter_pil(mask_array)
    gray_value = 128.0
    black_value = 0.0

    rows, cols = pil_array.shape
    for i in range(rows):
        for j in range(cols):
            if pil_array[i, j] == gray_value and numpy_array[i, j] == black_value:
                result_array[i, j] = black_value

    return Image.fromarray(result_array)


def filter_pil(pil_array):

    image = np.uint8(pil_array)
    gray_mask = (image == 128).astype(np.uint8)
    black_mask = (image == 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)

    for label in range(1, num_labels):
        label_mask = (labels == label).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated_label_mask = cv2.dilate(label_mask, kernel, iterations=1)

        intersection = cv2.bitwise_and(dilated_label_mask, black_mask)

        if np.any(intersection):
            image[labels == label] = 255

    return image
