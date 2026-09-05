import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
SEED = 42

TRAIN_DIR = "dataset/split_256/train"
VAL_DIR = "dataset/split_256/val"
TEST_DIR = "dataset/split_256/test"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# 3. PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# 4. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1)
])


# ============================================================
# 5. LOAD PRETRAINED VGG16
# ============================================================

print("\nLoading pretrained VGG16...")

base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze VGG16 initially
base_model.trainable = False


# ============================================================
# 6. BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# VGG16 expects its own preprocessing
x = tf.keras.applications.vgg16.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n================ MODEL SUMMARY ================\n")

model.summary()


# ============================================================
# 8. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-6
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/vgg16_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)


# ============================================================
# 9. TRAIN
# ============================================================

print("\n================ VGG16 TRAINING ================\n")

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

training_time = time.time() - start_time

print(
    f"\nTraining time: {training_time / 60:.2f} minutes"
)


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

model = tf.keras.models.load_model(
    "models/vgg16_best.keras"
)


# ============================================================
# 11. SAVE FINAL MODEL
# ============================================================

model.save(
    "models/vgg16.keras"
)

print("\nVGG16 model saved successfully.")


# ============================================================
# 12. TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("VGG16 - Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "vgg16_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VGG16 - Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "vgg16_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. TEST MODEL
# ============================================================

print("\n================ TESTING VGG16 ================\n")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Accuracy: {test_accuracy:.4f}"
)


# ============================================================
# 14. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

y_prob = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# ============================================================
# 15. METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n================ FINAL RESULTS ================\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n================ CLASSIFICATION REPORT ================\n")

print(report)

with open(
    RESULTS_DIR / "vgg16_classification_report.txt",
    "w"
) as f:
    f.write(report)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("VGG16 - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "vgg16_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["VGG16"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "Training_Time_Minutes": [training_time / 60]
})

results.to_csv(
    RESULTS_DIR / "vgg16_results.csv",
    index=False
)

print("\nResults saved to:")
print(RESULTS_DIR / "vgg16_results.csv")

print("\n================ VGG16 DONE ================\n")