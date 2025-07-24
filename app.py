from flask import Flask, request, render_template, session
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.secret_key = 'your_secret_key'

mp_face_mesh = mp.solutions.face_mesh

# 中文字數計算
def count_chinese(text):
    return len([c for c in text if '\u4e00' <= c <= '\u9fff'])

# 數字轉花色
def convert_to_symbol(num):
    mapping = {1: '♠', 2: '♥', 3: '♣', 4: '♦'}
    return mapping.get(num, str(num))

# 九宮格位置 + 牙齒判斷
def determine_position_and_teeth(image):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return "無法偵測臉"

        face = results.multi_face_landmarks[0]
        nose_tip = face.landmark[1]
        x = int(nose_tip.x * w)
        y = int(nose_tip.y * h)

        # 定位
        if y < cy and x < cx:
            pos = '左上'
        elif y < cy and x > cx:
            pos = '右上'
        elif y > cy and x < cx:
            pos = '左下'
        elif y > cy and x > cx:
            pos = '右下'
        elif abs(x - cx) <= w * 0.1:
            if y < cy:
                pos = '正上'
            elif y > cy:
                pos = '正下'
            else:
                pos = '正中'
        elif x < cx:
            pos = '正左'
        else:
            pos = '正右'

        # 牙齒判斷
        try:
            lip_top = face.landmark[13]
            lip_bottom = face.landmark[14]
            lip_distance = abs(lip_top.y - lip_bottom.y) * h
            has_teeth = lip_distance > 5
        except:
            has_teeth = False

        # 最終回傳編碼
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

    response = None

    if request.method == 'POST':
        # 使用者傳文字階段
        if session['step'] == 1 and 'text' in request.form:
            text = request.form['text'].strip()
            if '大師' in text:
                cleaned = text.replace('大師', '')
                count = count_chinese(cleaned)
                symbol = convert_to_symbol(count)
                session['symbol'] = symbol
                session['step'] = 2
                response = "你說什麼？"
            else:
                response = "？"

        elif session['step'] == 2 and 'text' in request.form:
            session['step'] = 3
            response = "拍個照片我看看～"

        # 使用者傳圖片階段
        elif session['step'] == 3 and 'image' in request.files:
            file = request.files['image']
            if file:
                img_array = np.frombuffer(file.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                number = determine_position_and_teeth(image)
                symbol = session.get('symbol', '?')
                response = f"{symbol}{number}"
                session.clear()

    return render_template('chat.html', response=response, step=session.get('step', 1))

if __name__ == '__main__':
    app.run(debug=True)
