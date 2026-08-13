"""
---------------------------------------------------------
VERINOTE

Module : Flask API

Purpose:
Provides REST API for counterfeit currency detection.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import sys
import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference.predict import predict


UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():

    return jsonify({
        "project": "VERINOTE",
        "status": "Running",
        "model": "MobileNetV2"
    })


@app.route("/predict", methods=["POST"])
def predict_note():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Only jpg, jpeg and png images are allowed."
        }), 400

    filename = secure_filename(file.filename)

    filepath = UPLOAD_FOLDER / filename

    file.save(filepath)

    result = predict(str(filepath))

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)