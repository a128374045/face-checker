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

# 載入人臉模型（不再使用嘴巴模型）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # 分成九宮格
        col = 0 if face_center_x < width / 3 else (1 if face_center_x < width * 2 / 3 else 2)
        row = 0 if face_center_y < height / 3 else (1 if face_center_y < height * 2 / 3 else 2)
        position = row * 3 + col + 1  # 九宮格 1~9

        # 不含牙齒偵測的對應表
        position_map = {
            1: 10,  # 左上
            2: 12,  # 正上
            3: 1,   # 右上
            4: 9,   # 正左
            5: 13,  # 中間
            6: 3,   # 正右
            7: 7,   # 左下
            8: 6,   # 正下
            9: 4,   # 右下
        }

        return str(position_map[position])

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
    app.run(debug=True)
