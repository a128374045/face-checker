from flask import Flask, request, render_template, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用於 session 記憶狀態

# 載入 Haar cascade 嘴巴模型（如果不使用牙齒偵測則可略過）
mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
if os.path.exists(mouth_cascade_path):
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
else:
    mouth_cascade = None  # 若檔案不存在就略過牙齒判斷

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# 判斷中文字數
def count_chinese(text):
    return len([ch for ch in text if '\u4e00' <= ch <= '\u9fff'])

# 花色對應轉換
def convert_to_symbol(count):
    mapping = {1: '♠', 2: '♥', 3: '♣', 4: '♦'}
    return mapping.get(count, str(count))

# 判斷九宮格位置與牙齒
def determine_position_and_teeth(image):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return "無法偵測到臉"

        face = results.multi_face_landmarks[0]

        # 中心點選鼻子 tip 位置
        nose_tip = face.landmark[1]
        x = int(nose_tip.x * w)
        y = int(nose_tip.y * h)

        pos = ''
        if y < cy and x < cx:
            pos = '左上'
        elif y < cy and x > cx:
            pos = '右上'
        elif y > cy and x < cx:
            pos = '左下'
        elif y > cy and x > cx:
            pos = '右下'
        elif abs(y - cy) < h * 0.1:
            if x < cx:
                pos = '正左'
            elif x > cx:
                pos = '正右'
            else:
                pos = '正中'
        elif y < cy:
            pos = '正上'
        elif y > cy:
            pos = '正下'

        # 嘴巴區域牙齒判斷（可簡化為上下唇距離）
        has_teeth = False
        try:
            top_lip = face.landmark[13]
            bottom_lip = face.landmark[14]
            lip_distance = abs((top_lip.y - bottom_lip.y) * h)
            has_teeth = lip_distance > 5
        except:
            pass

        # 九宮格編碼
        mapping = {
            '正中': 13, '正上': 12, '正下': 6,
            '正左': 9, '正右': 3,
            '左上': 10 if has_teeth else 11,
            '右上': 2 if has_teeth else 1,
            '左下': 7 if has_teeth else 8,
            '右下': 5 if has_teeth else 4
        }

        return str(mapping.get(pos, 0))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'step' not in session:
        session['step'] = 1

    response = ''
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form['text'].strip()
            if session['step'] == 1:
                if '大師' in text:
                    cleaned = text.replace('大師', '')
                    count = count_chinese(cleaned)
                    symbol = convert_to_symbol(count)
                    session['symbol'] = symbol
                    session['step'] = 2
                    response = '你說什麼？'
                else:
                    response = '？'
            elif session['step'] == 2:
                session['step'] = 3
                response = '拍個照片我看看吧～'
        elif 'image' in request.files:
            if session.get('step') == 3:
                file = request.files['image']
                img_array = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                number = determine_position_and_teeth(image)
                response = f"{session.get('symbol', '?')}{number}"
                session.clear()

    return render_template('chat.html', response=response, step=session['step'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
