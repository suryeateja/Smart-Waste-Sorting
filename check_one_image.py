import tensorflow as tf
import numpy as np

MODEL_PATH = "models/mobilenetv2.keras"

IMAGE_PATH = (
    "dataset/split_256/test/plastic/plastic_8.jpg"
)

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

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded!\n")


# Load image EXACTLY like the test dataset
image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

image = tf.keras.utils.img_to_array(image)

image = tf.expand_dims(
    image,
    axis=0
)


# Predict
prediction = model.predict(
    image,
    verbose=0
)[0]


# Show all classes
print("===================================")
print("Prediction for plastic_8.jpg")
print("===================================\n")

for i in np.argsort(prediction)[::-1]:
    print(
        f"{CLASS_NAMES[i]:12s} "
        f"{prediction[i] * 100:.2f}%"
    )


best = np.argmax(prediction)

print("\n-----------------------------------")
print("ACTUAL CLASS    : PLASTIC")
print(
    f"PREDICTED CLASS : "
    f"{CLASS_NAMES[best].upper()}"
)
print(
    f"CONFIDENCE      : "
    f"{prediction[best] * 100:.2f}%"
)
print("-----------------------------------")