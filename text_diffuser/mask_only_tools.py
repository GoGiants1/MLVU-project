import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def draw_centers_with_text(mask,centers,text_contents,font_size_list,coordinates,stroke):

    #input_img :numpy array
    #storke: 0-255 1차원 , 검정색 부분이 글자 ,
    #mask: 검정색 배경 하얀색 mask
    #centers: mask들의 중심좌표 

    image0 = np.ones_like(mask)
    image=Image.fromarray(image0)
    image=image.convert("L")
    image=np.array(image)

    # text가 있는 영역들이 회색으로 칠해진다.
    for w in range(image.shape[0]):
        for h in range(image.shape[1]):
            if mask[w,h]==1.0:
                image[w,h]=128.0
            else:
                continue


    image=Image.fromarray(image)
    image=image.convert("L")
    draw = ImageDraw.Draw(image)

    # choice list내에 있는 좌표와 가장 가까운 scene text 부분만 bear 로 대체가 된다.
    for chosen in coordinates:
        the_index=closest_index(choice_list=chosen,centers=centers)

        font = ImageFont.truetype("assets/font/Arial.ttf", font_size_list[the_index])
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
    image = cv2.imread("mask_sum.png")
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
        masks_size.append(size) #size is tuple (width, height)
        angle_list.append(angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
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
    print(masks_size)
    print(angle_list)
    exit()
    return (centers,masks_size,angle_list)



def fill_gray_with_text(pil_array, numpy_array):
    pil_array=np.array(pil_array)
    # PIL 이미지 배열과 numpy 배열의 크기가 같은지 확인
    assert pil_array.shape == numpy_array.shape, "The two arrays must have the same shape"

    result_array = pil_array.copy()
    pil_array=filter_pil(pil_array=pil_array)
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
