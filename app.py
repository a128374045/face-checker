from flask import Flask, render_template, request, session
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = "your_secret_key"

# 中文字判斷
def count_chinese_characters(text):
    return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')

# 花色轉換
def convert_to_suit(n):
    suits = ["♠️", "♥️", "♣️", "♦️"]
    return suits[(n - 1) % 4] if n > 0 else "？"

# 臉部與牙齒判斷
def detect_face_and_teeth(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(img)
    image_cv = open_cv_image[:, :, ::-1].copy()

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    height, width, _ = image_cv.shape
    h_third = height // 3
    w_third = width // 3

    if len(faces) == 0:
        return 0

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        col = center_x // w_third
        row = center_y // h_third
        index = row * 3 + col + 1

        # 模擬牙齒辨識
        mouth_region = gray[y + h // 2:y + h, x:x + w]
        white_pixels = cv2.countNonZero(mouth_region > 200)
        teeth = white_pixels > 50

        # 根據九宮格位置 + 牙齒
        if row == 0 and col == 0:
            return 11 if teeth else 10
        elif row == 0 and col == 2:
            return 2 if teeth else 1
        elif row == 2 and col == 0:
            return 8 if teeth else 7
        elif row == 2 and col == 2:
            return 5 if teeth else 4
        elif row == 0 and col == 1:
            return 12
        elif row == 2 and col == 1:
            return 6
        elif row == 1 and col == 0:
            return 9
        elif row == 1 and col == 2:
            return 3
        else:
            return 13

@app.route("/", methods=["GET", "POST"])
def index():
    if "step" not in session:
        session["step"] = 1

    response = ""
    step = session["step"]

    if request.method == "POST":
        if step == 1:
            text = request.form["text"]
            text_wo_keyword = text.replace("大師", "")
            count = count_chinese_characters(text_wo_keyword)
            session["suit"] = convert_to_suit(count)
            session["step"] = 2
            response = "什麼事？"

        elif step == 2:
            session["step"] = 3
            response = "拍個照片我看看"

        elif step == 3 and "image" in request.files:
            image = request.files["image"]
            image_bytes = image.read()
            number = detect_face_and_teeth(image_bytes)
            final_response = f"{session.get('suit', '？')}{number}"
            session.clear()
            response = final_response

    return render_template("index.html", step=session.get("step", 1), response=response)
