import os
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = "dataset/standardized_256"

classes = os.listdir(dataset_path)

plt.figure(figsize=(12, 8))

for i, class_name in enumerate(classes[:10]):

    class_path = os.path.join(dataset_path, class_name)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(class_path, images[0])

    image = Image.open(image_path)

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()