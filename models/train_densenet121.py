import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001

TRAIN_DIR = "dataset/split_256/train"
VAL_DIR = "dataset/split_256/val"
TEST_DIR = "dataset/split_256/test"

MODEL_PATH = "models/densenet121.keras"
BEST_MODEL_PATH = "models/densenet121_best.keras"

RESULTS_DIR = "results"

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

NUM_CLASSES = len(CLASS_NAMES)

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset loaded successfully!")


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ============================================================
# 4. BUILD DENSENET121 MODEL
# ============================================================

print("\nBuilding DenseNet121...")

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False


inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# DenseNet preprocessing
x = preprocess_input(x)

# Pretrained DenseNet121
x = base_model(x, training=False)

# Feature extraction
x = layers.GlobalAveragePooling2D()(x)

# Classification head
x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="DenseNet121_Waste_Classifier"
)


# ============================================================
# 5. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel created successfully!\n")
model.summary()


# ============================================================
# 6. CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 7. TRAIN
# ============================================================

print("\n========================================")
print("       TRAINING DENSENET121")
print("========================================\n")

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

training_time = time.time() - start_time

print(
    f"\nTraining completed in "
    f"{training_time / 60:.2f} minutes."
)


# ============================================================
# 8. SAVE FINAL MODEL
# ============================================================

model.save(MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")


# ============================================================
# 9. TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 10. CALCULATE METRICS
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


print("\n========================================")
print("          DENSENET121 RESULTS")
print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nPercentages:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

with open(
    f"{RESULTS_DIR}/densenet121_classification_report.txt",
    "w"
) as f:
    f.write(report)


# ============================================================
# 12. CONFUSION MATRIX
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
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DenseNet121 Confusion Matrix")

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/densenet121_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. ACCURACY GRAPH
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
plt.title("DenseNet121 Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/densenet121_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. LOSS GRAPH
# ============================================================

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
plt.title("DenseNet121 Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/densenet121_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. SAVE RESULTS CSV
# ============================================================

num_params = model.count_params()

model_size_mb = (
    os.path.getsize(MODEL_PATH)
    / (1024 * 1024)
)

results = pd.DataFrame([{
    "Model": "DenseNet121",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Parameters": num_params,
    "Model Size (MB)": model_size_mb,
    "Training Time (minutes)": training_time / 60
}])

results.to_csv(
    f"{RESULTS_DIR}/densenet121_results.csv",
    index=False
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("       DENSENET121 COMPLETED")
print("========================================")

print("\nSaved files:")

print("✓ models/densenet121.keras")
print("✓ models/densenet121_best.keras")
print("✓ results/densenet121_results.csv")
print("✓ results/densenet121_accuracy.png")
print("✓ results/densenet121_loss.png")
print("✓ results/densenet121_confusion_matrix.png")
print("✓ results/densenet121_classification_report.txt")

print("\nDone! 🎉")