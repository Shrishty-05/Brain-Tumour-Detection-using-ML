from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

MODEL_PATH = "best_brain_tumor_model.h5"
IMG_SIZE = (224, 224)

# Load model once
model = keras.models.load_model(MODEL_PATH)


def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = float(model.predict(img_array, verbose=0)[0][0])
    return pred


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded", confidence=None)

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected", confidence=None)

    try:
        img = Image.open(file).convert("RGB")
    except Exception:
        return render_template("index.html", prediction="Invalid image file", confidence=None)

    pred = predict_image(img)

    if pred > 0.5:
        confidence_value = pred * 100
        result = "Tumor Detected (YES)"
    else:
        confidence_value = (1 - pred) * 100
        result = "No Tumor (NO)"

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence_value:.2f}%",
        image_data=image_data
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=False)