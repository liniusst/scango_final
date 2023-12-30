from flask import Flask, jsonify, render_template, request
from flask_bcrypt import Bcrypt
import os
import sys

# sys.path.insert(0, "C:/ANPR/scango_final/")
from detection.app import detect_license_plate


app = Flask(__name__, static_url_path="/static")
app.config["SECRET_KEY"] = "4654f5dfadsrfasdr54e6rae"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "mp4", "avi"}

bcrypt = Bcrypt(app)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("index.html", result="No file part")

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        license_plate_detector = detect_license_plate(conf_score=0.75)
        ocr_result = license_plate_detector.process_video(file_path)

        return render_template("index.html", result=ocr_result)
    else:
        return render_template("index.html", result="Invalid file format")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
