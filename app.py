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

# 載入人臉與「微笑」（嘴巴）模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

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

        # 九宮格定位
        col = 0 if face_center_x < width / 3 else (1 if face_center_x < width * 2 / 3 else 2)
        row = 0 if face_center_y < height / 3 else (1 if face_center_y < height * 2 / 3 else 2)
        position = row * 3 + col + 1  # 1~9

        # 嘴巴偵測（改用 smile）
        roi_gray = gray[y + h//2:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        has_teeth = len(mouths) > 0

        # 對應回應數字
        position_map = {
            1: 11 if has_teeth else 10,
            2: 12,
            3: 2 if has_teeth else 1,
            4: 9,
            5: 13,
            6: 3,
            7: 8 if has_teeth else 7,
            8: 6,
            9: 5 if has_teeth else 4,
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
