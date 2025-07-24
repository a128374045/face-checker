from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 建立上傳資料夾
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 載入人臉與笑容（代替嘴巴）模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
if smile_cascade.empty():
    smile_cascade = None

def detect_position_and_teeth(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "沒找到臉"

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        height, width = img.shape[:2]

        # 判斷九宮格位置 (3x3)
        col = 0 if face_center_x < width / 3 else (1 if face_center_x < width * 2 / 3 else 2)
        row = 0 if face_center_y < height / 3 else (1 if face_center_y < height * 2 / 3 else 2)
        position = row * 3 + col + 1  # 1~9

        # 判斷笑容（有笑當作有牙齒）
        has_teeth = False
        if smile_cascade:
            roi_gray = gray[y + h//2:y + h, x:x + w]  # 臉下半部
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
            has_teeth = len(smiles) > 0

        # 九宮格位置對應輸出（依你的需求）
        position_map = {
            1: 11 if has_teeth else 10,  # 左上 有牙齒11 無牙齒10
            2: 12,                      # 正上
            3: 2 if has_teeth else 1,    # 右上 有牙齒2 無牙齒1
            4: 9,                       # 正左
            5: 13,                      # 中間
            6: 3,                       # 正右
            7: 7 if has_teeth else 8,    # 左下 有牙齒7 無牙齒8 （已交換）
            8: 6,                       # 正下
            9: 5 if has_teeth else 4    # 右下 有牙齒5 無牙齒4
        }

        return str(position_map.get(position, "偵測錯誤"))

    return "偵測失敗"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        if 'image' not in request.files:
            result = "未上傳圖片"
        else:
            file = request.files['image']
            if file.filename == '':
                result = "沒有選擇檔案"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                result = detect_position_and_teeth(filepath)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
