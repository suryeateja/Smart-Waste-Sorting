import os
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
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

TRAIN_DIR = "dataset/split_256/train"
VAL_DIR = "dataset/split_256/val"
TEST_DIR = "dataset/split_256/test"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("Using:", tf.config.list_physical_devices())


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nLoading test dataset...")

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

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# 3. PERFORMANCE SETTINGS
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# 4. CLASS WEIGHTS
# ============================================================

# Calculate the number of images in each training class

class_counts = []

for class_name in class_names:
    class_path = os.path.join(TRAIN_DIR, class_name)

    count = 0

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            count += 1

    class_counts.append(count)

print("\nTraining class counts:")

for name, count in zip(class_names, class_counts):
    print(f"{name:15s}: {count}")


# Create balanced class weights

labels = []

for class_index, count in enumerate(class_counts):
    labels.extend([class_index] * count)

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = {
    i: weight for i, weight in enumerate(weights)
}

print("\nClass weights:")

for i, weight in class_weights.items():
    print(f"{class_names[i]:15s}: {weight:.3f}")


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")


# ============================================================
# 6. BUILD CUSTOM CNN
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Data augmentation
    data_augmentation,

    # Normalize pixel values
    tf.keras.layers.Rescaling(1.0 / 255),

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # --------------------------------------------------------
    # Block 4
    # --------------------------------------------------------
    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    # 10 waste classes
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 7. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. DISPLAY MODEL
# ============================================================

print("\n================ MODEL SUMMARY ================\n")

model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n================ TRAINING ================\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "custom_cnn.keras"

model.save(model_path)

print("\nModel saved to:")
print(model_path)


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
plt.title("Custom CNN - Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "custom_cnn_accuracy.png",
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
plt.title("Custom CNN - Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    RESULTS_DIR / "custom_cnn_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. TEST SET PREDICTIONS
# ============================================================

print("\n================ TESTING ================\n")

y_true = []
y_pred = []

for images, labels_batch in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels_batch.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 14. CALCULATE METRICS
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
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n================ CLASSIFICATION REPORT ================\n")

print(report)


# Save classification report

with open(
    RESULTS_DIR / "custom_cnn_classification_report.txt",
    "w"
) as f:

    f.write(report)


# ============================================================
# 16. CONFUSION MATRIX
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
plt.title("Custom CNN - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "custom_cnn_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE NUMERICAL RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Custom CNN"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1]
})

results.to_csv(
    RESULTS_DIR / "custom_cnn_results.csv",
    index=False
)

print("\nResults saved to:")
print(RESULTS_DIR / "custom_cnn_results.csv")

print("\n================ DONE ================\n")