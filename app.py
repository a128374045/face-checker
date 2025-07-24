from flask import Flask, render_template, request, session, redirect, url_for
from face_utils import analyze_face_position
import os
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"

def count_chinese_characters(text):
    text = text.replace("大師", "")
    return sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')

def convert_to_symbol(num):
    symbols = ["♠️", "♥️", "♣️", "♦️"]
    return symbols[(num - 1) % 4] if num > 0 else "？"

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    step = session.get("step", 1)

    if request.method == "POST":
        if step == 1:
            text = request.form.get("text", "")
            chinese_count = count_chinese_characters(text)
            session["symbol"] = convert_to_symbol(chinese_count)
            session["step"] = 2
            response = "什麼事？"

        elif step == 2:
            session["step"] = 3
            response = "拍個照片我看看"

        elif step == 3:
            image = request.files["image"]
            if image:
                number = analyze_face_position(image)
                symbol = session.get("symbol", "？")
                session["step"] = 1  # 重置
                response = f"{symbol}{number if number else '？'}"
            else:
                response = "請重新上傳圖片"

    return render_template("index.html", response=response, step=session.get("step", 1))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
