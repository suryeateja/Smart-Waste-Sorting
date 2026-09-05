import os

dataset_path = "dataset/standardized_256"

total = 0

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"{class_name:15} : {len(images)}")
        total += len(images)

print("-" * 30)
print("TOTAL IMAGES:", total)