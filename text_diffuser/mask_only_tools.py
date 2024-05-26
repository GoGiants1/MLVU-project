import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def draw_centers_with_text(masks, center_ls, angle_ls, sample_text, font_size_ls, coordinates, stroke):

    #input_img :numpy array
    #storke: 0-255 1차원 , 검정색 부분이 글자 ,
    #mask: 검정색 배경 하얀색 mask
    #centers: mask들의 중심좌표 

    # text가 있는 영역들이 회색으로 칠해진다.
    
    """ 2. 검은색 배경에 흰색 마스크 """
    black_background = np.zeros((512, 512), dtype=np.uint8)
    
    for i in range(masks.shape[0]):
        black_background[np.where(masks[i] == 255)] = 255
    white_mask_img = black_background
    
    grey_masks_with_white_back = np.array(np.where(white_mask_img == 255, 128, 255), dtype=np.uint8)
    grey_masks_WB = Image.fromarray(grey_masks_with_white_back, "L")
    Image.fromarray(grey_masks_with_white_back).save("grey_masks_WB.png")
    
    # choice list내에 있는 좌표와 가장 가까운 scene text 부분만 bear 로 대체가 된다.
    for coord in coordinates:
        the_index = closest_index(coord, center_ls)
        
        # text stroke가 원래는 검정색이다. 그러므로 이걸 흰색으로 바꿔주는 작업.
        print(f"masks[the_index].shape: {masks[the_index].shape}")
        stroke[np.where(masks[the_index]==255)] = 255 # mask에 해당하는 text stroke를 지워준다
        
        size_tuple = font_size_ls[the_index]
        w,h = size_tuple
        font_size = min(w,h)
        
        font = ImageFont.truetype("assets/font/Arial.ttf", font_size)
        _, _, text_w, text_h = font.getbbox(sample_text)
        while max(text_w, text_h) > max(w, h):
            font_size -= 1
            font = ImageFont.truetype("assets/font/Arial.ttf", font_size)
            _, _, text_w, text_h = font.getbbox(sample_text)
        
        # 512, 512 이미지 중앙에 텍스트를 그림
        text_image = Image.new('L', (512, 512), 255)
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text(((512-text_w)/2, (512-text_h)/2), sample_text, font=font, fill=0)
        Image.fromarray(np.array(text_image)).save("text_image.png")
        
        # 512, 512 이미지 중앙에 마스킹용 텍스트를 그림 (이거 없으면 잘 안되더라구요)
        mask_image = Image.new('L', (512, 512))
        mask_draw = ImageDraw.Draw(mask_image)
        mask_draw.text(((512-text_w)/2, (512-text_h)/2), sample_text, font=font, fill=255)
                
        # 둘 다 회전. 이미지 중앙을 기준으로 회전한 것이라, 제자리에서 잘 회전합니다.
        mask_rotate = mask_image.rotate(angle_ls[the_index], expand=1)
        text_rotate = text_image.rotate(angle_ls[the_index], expand=1)
        Image.fromarray(np.array(text_rotate)).save("text_rotate.png")
        
        # 글씨를 이제 paste하는데 center_ls[the_index]가 중심이 되도록 paste한다.
        print(f"center_ls[the_index]: {center_ls[the_index]}")
        print(f"center_ls[the_index][0] - 512/2: {center_ls[the_index][0]}")
        grey_masks_WB.paste(text_rotate, (int(center_ls[the_index][0] - 512/2), int(center_ls[the_index][1] - 512/2)), mask_rotate)
        grey_masks_WB.save("grey_masks_WB_text.png")
    
    grey_masks_WB_array = np.array(grey_masks_WB)
    grey_masks_WB_array[np.where(stroke == 0)] = 0
    
    Image.fromarray(grey_masks_WB_array).save("grey_masks_WB_stroke.png")
    return grey_masks_WB_array

def closest_index(choice, center_ls):
    min_distance = float('inf')
    min_index = -1

    for i, center in enumerate(center_ls):
        distance = ((choice[0] - center[0])**2 + (choice[1] - center[1])**2)**0.5

        if distance < min_distance:
            min_distance = distance
            min_index = i

    return min_index


def sorting_coord(list):

    # 중심점들의 합을 기준으로 오름차순 정렬
    center_ls_with_sums = [(sum(tup), tup) for tup in list]
    sorted_centers_with_sums = sorted(center_ls_with_sums)
    sorted_centers = [tup for _, tup in sorted_centers_with_sums]

    return sorted_centers


def take_info(masks):
    #np_image는 검은 바탕에 흰색 박스들로 이루어진 이미지
    #_, binary_image = cv2.threshold(np_image, 254, 255, cv2.THRESH_BINARY)
    masks_size_ls = []
    centers_ls = []
    angle_ls = []
     
    for i in range(masks.shape[0]):
        # 픽셀 값이 50보다 작으면 0으로, 50보다 크면 255로 이진화
        _, binary = cv2.threshold(masks[i], 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 이진화된 이미지에서 가장 바깥쪽 외곽선을 찾음
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 외곽선들이 여러 개 나오는거라, 그 중 첫번째 외곽선을 저장
        contour = contours[0]
        
        # 외곽선들 중 가장 큰 외곽선을 저장하기 위한 for 문
        if len(contours) > 1:
            for tmp in contours:
                if cv2.contourArea(tmp) > cv2.contourArea(contour):
                    contour = tmp
                    
        # 외곽선을 감싸는 최소 사각형을 구하고, 중심점 사이즈 각도를 저장
        rect = cv2.minAreaRect(contour)
        center, size, angle = rect
        
        # 검은 배경의 이미지를 생성
        mask = np.zeros((512, 512), dtype=np.uint8)

        # 사각형의 네 꼭짓점을 계산
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 사각형을 흰색으로 그림
        cv2.drawContours(mask, [box], 0, (255), thickness=cv2.FILLED)

        # NumPy 배열을 PIL 이미지로 변환
        mask = Image.fromarray(mask)

        # PNG 파일로 저장
        mask.save(f"debugging_image/rectangle_{i}.png")
        
        # 중심점 좌표와 사이즈를 저장
        centers_ls.append(center) #center is tuple (x, y) 
        masks_size_ls.append((size[0], size[1])) #size is tuple 회전 전혀 안 했을때의 (직사각형의 높이, 직사각형의 너비)
        
        # 사각형의 너비가 높이보다 크면 회전 각도를 조정
        if size[1] > size[0]:
            angle = 90 - angle
        angle_ls.append(angle)
    return (centers_ls, masks_size_ls, angle_ls)



def fill_gray_with_text(white_back_GS, stroke_array):
    
    # PIL 이미지 배열과 numpy 배열의 크기가 같은지 확인
    assert white_back_GS.shape == stroke_array.shape, "The two arrays must have the same shape"

    result_array = white_back_GS.copy()
    white_back_GS = filter_pil(white_back_GS)

    rows, cols = white_back_GS.shape
    for i in range(rows):
        for j in range(cols):
            # 회색 배경에서 stroke_array가 0이라면, 결과 배열에도 0을 저장해 검은색 글씨가 잘 나오도록
            if white_back_GS[i, j] == 128.0 and stroke_array[i, j] == 0.0:
                result_array[i, j] = 0.0

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
