from flask import Flask, jsonify
from flask_bcrypt import Bcrypt
from flask import flash, redirect, render_template, request, url_for
import os


app = Flask(__name__, static_url_path='/static')
app.config["SECRET_KEY"] = "4654f5dfadsrfasdr54e6rae"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

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

        # ocr_result = run_ocr(file_path)
        ocr_result = None
        return render_template("index.html", result=ocr_result)
    else:
        return render_template("index.html", result="Invalid file format")
    
# def run_ocr(image_path):
#     try:
#         detection = detect_license_plate(image_path)
#         result = detection.final_dict()
#         return result
#     except Exception as e:
#         return str(e)


if __name__ == "__main__":
    app.run(debug=True, port=5000)