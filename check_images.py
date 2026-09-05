from PIL import Image
import os

dataset_path = "dataset/standardized_256"

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:
        image_path = os.path.join(class_path, images[0])

        image = Image.open(image_path)

        print(class_name, "→", image.size, image.mode)