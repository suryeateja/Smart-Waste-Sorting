import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/mobilenetv2.keras"

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

print("Loading MobileNetV2...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!\n")


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:5]

    print("\n====================================")
    print("IMAGE:", image_path)
    print("====================================")

    for i, index in enumerate(top_indices, start=1):
        print(
            f"{i}. {CLASS_NAMES[index]:12s} "
            f"{predictions[index] * 100:.2f}%"
        )


# ------------------------------------------------
# CHANGE THESE PATHS
# ------------------------------------------------

images = [
    r"C:\Users\surya\OneDrive\Desktop\Smart-Waste-Sorting\dataset\split_256\test\glass\glass_6.jpg",

    r"C:\Users\surya\OneDrive\Desktop\Smart-Waste-Sorting\dataset\split_256\test\metal\metal_8.jpg",

    r"C:\Users\surya\OneDrive\Desktop\Smart-Waste-Sorting\dataset\split_256\test\plastic\plastic_8.jpg"
]


for image in images:

    if os.path.exists(image):
        predict_image(image)
    else:
        print("\nNOT FOUND:")
        print(image)