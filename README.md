# ♻️ Vision-Based Waste Classification System

### Deep Learning Based Multi-Class Waste Classification for Efficient Recyclable Material Sorting

A deep learning-based computer vision system that classifies waste images into **10 different waste categories** using multiple CNN architectures and transfer learning models.

The project evaluates **six different deep learning models** on the same dataset and compares their classification performance to study the trade-off between **accuracy, model architecture, and deployment efficiency**.

---

## 📌 Project Overview

Manual waste segregation is time-consuming, labour-intensive, and prone to human error.

This project explores the use of **Deep Learning and Computer Vision** for automated waste classification.

Given an input image, the system predicts the waste category from the following 10 classes:

- 🔋 Battery
- 🌱 Biological
- 📦 Cardboard
- 👕 Clothes
- 🪟 Glass
- 🔩 Metal
- 📄 Paper
- 🧴 Plastic
- 👟 Shoes
- 🗑️ Trash

The project compares models ranging from a CNN trained from scratch to modern transfer-learning architectures.

---

## 🎯 Objectives

- Develop a vision-based waste classification system.
- Classify waste into 10 different categories.
- Compare CNN architectures with different computational characteristics.
- Study the effect of transfer learning on classification performance.
- Evaluate models using Accuracy, Precision, Recall and F1-score.
- Identify a model suitable for practical deployment.
- Develop a web-based interface for image-based prediction.

---

## 📊 Dataset

### Garbage Classification V2

The project uses the **Garbage Classification V2** dataset.

| Property | Details |
|---|---|
| Total Images | 12,259 |
| Number of Classes | 10 |
| Image Type | RGB |
| Standardized Image Size | 256 × 256 |
| Model Input Size | 224 × 224 |
| Training Images | 8,578 |
| Validation Images | 1,833 |
| Test Images | 1,848 |

### Dataset Classes

```text
battery
biological
cardboard
clothes
glass
metal
paper


Dataset Split
The dataset was divided into:
70% → Training
15% → Validation
15% → Testing

A fixed random seed was used to maintain a consistent split across the experiments.
The dataset is not included in this repository.

🧠 Models
Six different CNN architectures were evaluated.
Phase 1
1. Custom CNN
2. VGG16
3. MobileNetV2
Phase 2
4. DenseNet121
5. EfficientNetB0
6. VGG19
🏗️ Model Details
1. Custom CNN
A CNN architecture designed and trained from scratch.
Main components:
Input Image
     ↓
Data Augmentation
     ↓
Conv2D + Batch Normalization + MaxPooling
     ↓
Conv2D + Batch Normalization + MaxPooling
     ↓
Conv2D + Batch Normalization + MaxPooling
     ↓
Conv2D + Batch Normalization + MaxPooling
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Dropout
     ↓
Softmax

The Custom CNN was used as a baseline model to compare training from scratch against transfer learning.
2. VGG16
VGG16 pretrained on ImageNet was used as a feature extractor.
The convolutional base was frozen and a custom classification head was added for the 10 waste classes.
3. MobileNetV2
MobileNetV2 pretrained on ImageNet was used through transfer learning.
MobileNetV2 was included because of its computational efficiency and suitability for resource-constrained or edge-oriented applications.
4. DenseNet121
DenseNet121 was evaluated as a deeper transfer-learning architecture.
Dense connections allow feature information from earlier layers to be reused by later layers.
5. EfficientNetB0
EfficientNetB0 was evaluated to study the balance between classification accuracy and computational efficiency.
6. VGG19
VGG19 was included as a deeper VGG-family architecture to compare its performance with VGG16.
📈 Experimental Results
All six models were evaluated on the same unseen test set.
Model	Accuracy	Precision	Recall	F1-Score
Custom CNN	58.87%	61.65%	58.87%	57.80%
VGG16	86.90%	86.89%	86.90%	86.79%
MobileNetV2	87.93%	88.20%	87.93%	87.81%
DenseNet121	89.77%	89.85%	89.77%	89.72%
EfficientNetB0	91.18%	91.29%	91.18%	91.15%
VGG19	85.82%	85.95%	85.82%	85.69%


Key Observation
EfficientNetB0 achieved the highest classification performance among the six evaluated models:
Accuracy: 91.18%
F1-Score: 91.15%
The results also show a substantial improvement from the Custom CNN baseline to transfer-learning architectures.
Custom CNN       → 58.87%
VGG16            → 86.90%
MobileNetV2      → 87.93%
DenseNet121      → 89.77%
EfficientNetB0   → 91.18%
VGG19            → 85.82%

🔬 Methodology
                Dataset
                   │
                   ▼
        Image Preprocessing
                   │
                   ▼
        Data Augmentation
                   │
                   ▼
        ┌─────────────────────┐
        │   Six CNN Models    │
        └─────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Custom       VGG16      MobileNetV2
       │           │           │
       └───────────┼───────────┘
                   │
             Phase 2 Models
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  DenseNet121  EfficientNetB0  VGG19
                   │
                   ▼
             Model Evaluation
                   │
                   ▼
        Accuracy / Precision /
        Recall / F1-Score
                   │
                   ▼
           Model Comparison
                   │
                   ▼
          Prediction Web App

🖥️ Web Application
A Flask-based web interface was developed for testing the trained model.
The application allows the user to:
- Upload an image.
- Preview the uploaded image.
- Run waste classification.
- View the predicted waste category.
- View prediction confidence.
- View the top 3 predictions.
- View basic model information.
Current Prediction Model
The web application uses:
EfficientNetB0

The model achieved:
Accuracy → 91.18%
Classes  → 10
Dataset  → 12,259 images

📂 Project Structure
Smart-Waste-Sorting/
│
├── dataset/
│   └── ...
│
├── models/
│   ├── train_custom_cnn.py
│   ├── train_vgg16.py
│   ├── train_mobilenetv2.py
│   ├── train_densenet121.py
│   ├── train_efficientnetb0.py
│   ├── train_vgg19.py
│   │
│   ├── custom_cnn.keras
│   ├── vgg16.keras
│   ├── mobilenetv2.keras
│   ├── densenet121.keras
│   ├── efficientnetb0.keras
│   └── vgg19.keras
│
├── results/
│   ├── *_accuracy.png
│   ├── *_loss.png
│   ├── *_confusion_matrix.png
│   ├── *_classification_report.txt
│   └── *_results.csv
│
├── webapp/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── style.css
│   │   └── script.js
│   └── uploads/
│
├── .gitignore
└── README.md

⚙️ Technologies Used
Programming
- Python
Deep Learning
- TensorFlow
- Keras
- CNN
- Transfer Learning
Data Processing
- NumPy
- Pandas
- Pillow
- Scikit-learn
Visualization
- Matplotlib
- Seaborn
Web Application
- Flask
- HTML
- CSS
- JavaScript
Development Tools
- Jupyter Notebook
- Git
- GitHub
- VS Code
🚀 Installation
1. Clone the Repository
git clone https://github.com/suryaeteja/Smart-Waste-Sorting.git
cd Smart-Waste-Sorting

2. Create a Virtual Environment
python -m venv .venv

Activate it on Windows:
.venv\Scripts\activate

3. Install Dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow flask split-folders

▶️ Running the Web Application
Navigate to the web application folder:
cd webapp

Run:
python app.py

Then open the local URL displayed in the terminal, usually:
http://127.0.0.1:5000

Upload an image and click Analyze Waste.
🧪 Model Training
Each model has its own training script inside the models/ directory.
For example:
python models/train_custom_cnn.py

python models/train_vgg16.py

python models/train_mobilenetv2.py

python models/train_densenet121.py

python models/train_efficientnetb0.py

python models/train_vgg19.py

The scripts train the models and save the trained .keras files and evaluation results.
📊 Evaluation Metrics
The models are evaluated using:
Accuracy
Measures the overall percentage of correctly classified images.
Precision
Measures how many predictions for a class were actually correct.
Recall
Measures how many actual samples of a class were correctly detected.
F1-Score
Harmonic mean of precision and recall.
Confusion Matrix
Used to analyze class-wise prediction errors and identify commonly confused waste categories.
🔍 Research Contribution
The project does not introduce a new CNN architecture.
Instead, the main contribution is a systematic experimental comparison of different CNN architectures on a common 10-class waste classification problem.
The study compares:
- A CNN trained from scratch.
- Classical transfer-learning architecture.
- Lightweight architecture.
- Dense connectivity architecture.
- Efficient architecture.
- Deeper VGG architecture.
This allows the project to investigate the relationship between:
Classification Performance
          +
Model Architecture
          +
Computational Efficiency
          +
Deployment Suitability

⚠️ Limitations
The current system performs image classification, meaning it predicts one main waste category for an input image.
Current limitations include:
- Multiple objects in a single image may be difficult to classify.
- Complex backgrounds can affect predictions.
- Lighting and image quality can influence classification.
- The current system does not perform object detection or instance segmentation.
- Classification category does not universally determine recyclability because recycling rules can vary by location and facility.
🔮 Future Work
Possible future improvements include:
- Object detection for multiple waste objects.
- Real-world image testing under different lighting conditions.
- Deployment on edge/mobile devices.
- Model compression and quantization.
- Grad-CAM / Score-CAM based explainability.
- Confidence-based unknown/uncertain prediction handling.
- Integration with automated physical sorting systems.
- Evaluation using model size and inference time for deployment analysis.
👨‍💻 Team
Team D13
Deep Learning Project
Course: 23CSE473
📚 References
The project was developed with reference to research work on:
- Waste classification using CNNs.
- Transfer learning for waste image classification.
- Lightweight CNN architectures.
- DenseNet and EfficientNet based image classification.
- Real-time waste classification systems.
⭐ Project Status
Phase 1        ✅ Complete
Phase 2        ✅ Complete
Six Models     ✅ Evaluated
Model Analysis ✅ Complete
Web Application ✅ Complete

Current Best Classification Result
Model:       EfficientNetB0
Accuracy:    91.18%
F1-Score:    91.15%
Classes:     10
Dataset:     12,259 images

♻️ Smart Waste Sorting
Deep Learning • Computer Vision • Transfer Learning • Waste Classification

### One important thing before you commit it

Your actual repository excludes the dataset through `.gitignore`, so **don't include a claim that users can immediately train the models after cloning unless they have the dataset**.

Also, your `.keras` files are large, so the README is fine as-is, but later you can add a **“Model Downloads”** section if you decide to move the model files to Git LFS or another storage solution.
plastic
shoes
trash
