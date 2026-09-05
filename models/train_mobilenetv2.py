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


# ============================================
# LOAD MODEL
# ============================================

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================
# GET IMAGE
# ============================================

image_path = input(
    "\nEnter the path of your image: "
).strip().strip('"')


try:
    image = Image.open(image_path).convert("RGB")

except Exception as e:
    print("\nERROR: Could not open image.")
    print(e)
    exit()


# ============================================
# PREPARE IMAGE
# ============================================

image = image.resize(IMG_SIZE)

image_array = np.array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================
# PREDICTION
# ============================================

# IMPORTANT:
# DO NOT use preprocess_input here!
#
# The MobileNetV2 preprocessing is already
# inside the trained model.

predictions = model.predict(
    image_array,
    verbose=0
)[0]


# ============================================
# TOP 3 PREDICTIONS
# ============================================

top_indices = np.argsort(
    predictions
)[::-1][:3]


print("\n========================================")
print("       WASTE CLASSIFICATION")
print("          MobileNetV2")
print("========================================")

print("\nTop 3 predictions:\n")

for rank, index in enumerate(
    top_indices,
    start=1
):

    class_name = CLASS_NAMES[index]

    confidence = predictions[index] * 100

    print(
        f"{rank}. {class_name:<12} "
        f"{confidence:.2f}%"
    )


# ============================================
# FINAL PREDICTION
# ============================================

best_index = np.argmax(predictions)

best_class = CLASS_NAMES[best_index]

best_confidence = predictions[best_index] * 100


print("\n----------------------------------------")

print(
    f"FINAL PREDICTION : "
    f"{best_class.upper()}"
)

print(
    f"CONFIDENCE       : "
    f"{best_confidence:.2f}%"
)

print("----------------------------------------")