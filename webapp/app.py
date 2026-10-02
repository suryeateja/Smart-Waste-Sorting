import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image


# ============================================================
# APP SETTINGS
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "../models/efficientnetb0.keras"

print("Loading EfficientNetB0...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]


IMG_SIZE = (224, 224)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        }), 400

    try:

        # Open image
        image = Image.open(file).convert("RGB")

        # Resize
        image = image.resize(IMG_SIZE)

        # Convert to NumPy
        image_array = np.array(image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # EfficientNetB0 handles its own
        # input rescaling internally.
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Get top 3
        top_indices = np.argsort(
            predictions
        )[::-1][:3]

        top_predictions = []

        for index in top_indices:

            top_predictions.append({
                "class": CLASS_NAMES[index],
                "confidence": round(
                    float(predictions[index] * 100),
                    2
                )
            })

        best_index = np.argmax(predictions)

        result = {
            "prediction": CLASS_NAMES[best_index],
            "confidence": round(
                float(predictions[best_index] * 100),
                2
            ),
            "top_predictions": top_predictions
        }

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )