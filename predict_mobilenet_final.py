import os
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = os.path.abspath("models/mobilenetv2.keras")

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


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("       MOBILE NET V2 DIAGNOSTIC")
print("==============================================")

print("\nModel path:")
print(MODEL_PATH)

print("\nDoes model file exist?")
print(os.path.exists(MODEL_PATH))

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

print("\nModel input:")
print(model.input_shape)

print("\nModel output:")
print(model.output_shape)


# ============================================================
# SHOW IMPORTANT MODEL LAYERS
# ============================================================

print("\n==============================================")
print("MODEL LAYERS")
print("==============================================")

for i, layer in enumerate(model.layers):

    print(i, layer.name, type(layer).__name__)


# ============================================================
# GET IMAGE
# ============================================================

image_path = input(
    "\nEnter image path: "
).strip().strip('"')


print("\nImage path actually used:")
print(image_path)

print("\nDoes image exist?")
print(os.path.exists(image_path))


# ============================================================
# LOAD IMAGE
# ============================================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

image = tf.keras.utils.img_to_array(image)

print("\nImage shape:")
print(image.shape)

print("\nPixel range BEFORE model:")
print(
    "min =",
    image.min(),
    "max =",
    image.max()
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image = tf.expand_dims(
    image,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

# IMPORTANT:
#
# DO NOT preprocess the image here.
#
# Your training code puts MobileNetV2 preprocessing
# INSIDE the model.
#
# Therefore we give the model RAW pixel values.

predictions = model.predict(
    image,
    verbose=0
)[0]


# ============================================================
# RESULTS
# ============================================================

print("\n==============================================")
print("             PREDICTIONS")
print("==============================================")

top_indices = np.argsort(
    predictions
)[::-1]


for rank, index in enumerate(
    top_indices[:10],
    start=1
):

    print(
        f"{rank}. "
        f"{CLASS_NAMES[index]:12s} "
        f"{predictions[index] * 100:.2f}%"
    )


# ============================================================
# FINAL
# ============================================================

best_index = np.argmax(predictions)

print("\n==============================================")

print(
    "FINAL PREDICTION :",
    CLASS_NAMES[best_index].upper()
)

print(
    "CONFIDENCE       :",
    f"{predictions[best_index] * 100:.2f}%"
)

print("==============================================")