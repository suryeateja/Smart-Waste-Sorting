import splitfolders

input_folder = "dataset/standardized_256"
output_folder = "dataset/split_256"

splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.70, 0.15, 0.15)
)

print("Dataset split completed!")