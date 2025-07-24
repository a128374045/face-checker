import cv2
import numpy as np
import face_recognition

def analyze_face_position(image_stream):
    image = face_recognition.load_image_file(image_stream)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return None

    top, right, bottom, left = face_locations[0]
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    h, w, _ = image.shape
    grid_x = w // 3
    grid_y = h // 3

    col = center_x // grid_x
    row = center_y // grid_y

    position_map = {
        (0, 0): '左上', (0, 1): '正上', (0, 2): '右上',
        (1, 0): '正左', (1, 1): '中間', (1, 2): '正右',
        (2, 0): '左下', (2, 1): '正下', (2, 2): '右下',
    }

    position = position_map.get((row, col), '未知')

    # 嘴巴區域截取
    face_landmarks = face_recognition.face_landmarks(image)
    if not face_landmarks:
        return '未知'

    mouth = face_landmarks[0].get('top_lip', [])
    if mouth:
        x_coords = [point[0] for point in mouth]
        y_coords = [point[1] for point in mouth]
        mouth_img = image[min(y_coords):max(y_coords), min(x_coords):max(x_coords)]
        gray = cv2.cvtColor(mouth_img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        has_teeth = white_pixels > 100
    else:
        has_teeth = False

    # 對應回傳數字
    table = {
        '中間': 13, '正上': 12, '正下': 6,
        '正左': 9, '正右': 3,
        '左上': 11 if has_teeth else 10,
        '右上': 2 if has_teeth else 1,
        '左下': 8 if has_teeth else 7,
        '右下': 5 if has_teeth else 4,
    }

    return table.get(position, 0)
