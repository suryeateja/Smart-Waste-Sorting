import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/vgg16.keras"

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
# LOAD MODEL
# ============================================================

print("\nLoading VGG16 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# GET IMAGE PATH
# ============================================================

image_path = input(
    "\nEnter the path of your image: "
).strip().strip('"')


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(image_path).convert("RGB")

except Exception as e:

    print("\nERROR: Could not open image.")
    print(e)
    exit()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = image.resize(IMG_SIZE)

image_array = np.array(image)

# Add batch dimension
image_array = np.expand_dims(
    image_array,
    axis=0
)

# VGG16 preprocessing
image_array = tf.keras.applications.vgg16.preprocess_input(
    image_array
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    image_array,
    verbose=0
)[0]


# ============================================================
# GET TOP PREDICTIONS
# ============================================================

top_indices = np.argsort(
    predictions
)[::-1][:3]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("          WASTE CLASSIFICATION")
print("========================================")

print("\nTop 3 predictions:\n")

for rank, index in enumerate(top_indices, start=1):

    class_name = CLASS_NAMES[index]

    confidence = predictions[index] * 100

    print(
        f"{rank}. {class_name:<12} "
        f"{confidence:.2f}%"
    )


best_index = np.argmax(predictions)

best_class = CLASS_NAMES[best_index]

best_confidence = predictions[best_index] * 100


print("\n----------------------------------------")

print(
    f"FINAL PREDICTION : {best_class.upper()}"
)

print(
    f"CONFIDENCE       : {best_confidence:.2f}%"
)

print("----------------------------------------")