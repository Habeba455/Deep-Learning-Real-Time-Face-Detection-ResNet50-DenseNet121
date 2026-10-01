# Real-Time Real vs. Fake Face Detection with Hybrid CNN
### ResNet50 + DenseNet121 | Deep Learning | Computer Vision | TensorFlow | OpenCV

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/ResNet50-Feature%20Extractor-blue">
  <img src="https://img.shields.io/badge/DenseNet121-Feature%20Extractor-green">
  <img src="https://img.shields.io/badge/Test%20Accuracy-96.94%25-brightgreen">
</p>

---

## Overview

This project implements an end-to-end computer vision system for **face localization and Real vs. Fake face classification** using a custom **Hybrid Convolutional Neural Network** built on top of **ResNet50 and DenseNet121**.

Rather than relying on a single CNN backbone, the system extracts complementary deep visual representations from two ImageNet-pretrained networks and combines them through **feature-level fusion** before classification.

The complete pipeline covers:

- Face localization
- Face ROI extraction
- Image preprocessing
- Transfer learning
- Dual-backbone feature extraction
- Deep feature fusion
- Binary classification
- Confidence-based prediction
- Model evaluation
- Model persistence

The model was trained and evaluated using a large-scale dataset containing **140,000 Real and Fake facial images**, reaching approximately **96.94% accuracy on 20,000 unseen test images**.

---

## Key Results

| Metric | Result |
|---|---:|
| Total Dataset | **140,000 images** |
| Training Set | **100,000 images** |
| Validation Set | **20,000 images** |
| Test Set | **20,000 images** |
| Number of Classes | **2 — Real / Fake** |
| Input Resolution | **256 × 256 × 3** |
| Training Accuracy | **~97.56%** |
| Validation Accuracy | **~97.07%** |
| Test Accuracy | **~96.94%** |
| Validation Loss | **~0.0786** |
| Test Loss | **~0.0797** |
| CNN Backbones | **ResNet50 + DenseNet121** |

---

## Problem Statement

Distinguishing authentic facial images from artificially generated or manipulated faces is a challenging computer vision problem.

Real and fake faces may share almost identical high-level characteristics, while the discriminative information can exist in subtle differences involving:

- Texture
- Local facial structure
- Edge consistency
- Spatial patterns
- Fine visual artifacts
- High-level semantic representations

A single CNN architecture may not capture all of these patterns equally well.

This project therefore uses a **dual-backbone architecture**, allowing two different deep neural networks to independently analyze the same facial image before combining their learned representations.

---

## System Architecture

The proposed system combines **ResNet50** and **DenseNet121** into a unified deep-learning pipeline.

```text
                       Input Image
                            │
                            ▼
                     Face Detection
                            │
                            ▼
                    Face ROI Extraction
                            │
                            ▼
                  Resize + Normalization
                            │
                            ▼
                     256 × 256 × 3
                            │
               ┌────────────┴────────────┐
               │                         │
               ▼                         ▼
           ResNet50                 DenseNet121
        ImageNet Weights          ImageNet Weights
               │                         │
               ▼                         ▼
       Deep Feature Maps         Deep Feature Maps
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                  Feature Concatenation
                            │
                            ▼
                  Global Average Pooling
                            │
                            ▼
                    Dense Layer (256)
                            │
                            ▼
                           ReLU
                            │
                            ▼
                       Dropout 0.3
                            │
                            ▼
                         Softmax
                            │
                   ┌────────┴────────┐
                   │                 │
                   ▼                 ▼
                 REAL               FAKE
```

---

## Why a Hybrid CNN?

The central idea behind this project is that different CNN architectures learn different representations of the same image.

Instead of selecting only one backbone, the system combines the strengths of both architectures.

### ResNet50

ResNet50 uses **residual connections** to enable efficient training of deep neural networks.

Residual learning helps information and gradients propagate through many convolutional layers, enabling the network to learn rich hierarchical visual features.

It is particularly effective at extracting high-level structural representations from images.

### DenseNet121

DenseNet121 introduces **dense connectivity**, where layers receive feature information from preceding layers.

This encourages:

- Feature reuse
- Efficient information propagation
- Strong gradient flow
- Preservation of lower-level visual information

### Feature Fusion

The representations generated by both networks are combined rather than making independent final predictions.

Conceptually:

```text
F_resnet = ResNet50(x)

F_densenet = DenseNet121(x)

F_hybrid = Concatenate(
    F_resnet,
    F_densenet
)

Prediction = Classifier(F_hybrid)
```

This creates a richer joint feature representation for the final Real/Fake classification stage.

---

## Dataset

The model was trained and evaluated on a large-scale dataset containing **140,000 facial images**.

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 100,000 |
| Validation | 20,000 |
| Testing | 20,000 |
| **Total** | **140,000** |

The classification problem contains two classes:

```text
Real
Fake
```

A typical dataset organization follows:

```text
Dataset/
│
├── Train/
│   ├── Real/
│   └── Fake/
│
├── Validation/
│   ├── Real/
│   └── Fake/
│
└── Test/
    ├── Real/
    └── Fake/
```

---

## Image Preprocessing Pipeline

Before an image reaches the classifier, it passes through a preprocessing pipeline designed to produce consistent input for the neural network.

```text
Raw Image
    │
    ▼
Face Localization
    │
    ▼
Bounding Box
    │
    ▼
Face ROI Extraction
    │
    ▼
Resize to 256 × 256
    │
    ▼
Pixel Normalization
    │
    ▼
Tensor Preparation
    │
    ▼
Hybrid CNN
```

The model expects RGB images with the shape:

```python
(256, 256, 3)
```

Images are loaded in batches of:

```python
batch_size = 32
```

Pixel values are normalized before being processed by the neural network.

---

## Transfer Learning

Both backbone networks use pretrained **ImageNet weights**.

Instead of learning basic visual representations entirely from scratch, transfer learning allows the system to start with representations learned from a large and diverse image dataset.

The pretrained networks already contain useful representations for visual characteristics such as:

```text
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Object Parts
  ↓
High-Level Visual Features
```

These representations are then adapted to the Real/Fake facial classification problem.

---

## Feature-Level Fusion

The most important architectural component is the fusion of the two CNN branches.

For every input image:

```text
                    Input
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
         ResNet50           DenseNet121
            │                   │
            ▼                   ▼
      Feature Tensor       Feature Tensor
            │                   │
            └─────────┬─────────┘
                      │
                      ▼
                  Concatenate
                      │
                      ▼
              Hybrid Representation
                      │
                      ▼
             Global Average Pooling
                      │
                      ▼
                  Classifier
```

This approach allows the final classifier to operate on information learned by **both CNN families** instead of relying on a single representation.

---

## Classification Head

After feature fusion, the combined representation passes through the classification stage.

The classification head contains:

```text
Feature Concatenation
        │
        ▼
Global Average Pooling
        │
        ▼
Dense — 256 Units
        │
        ▼
       ReLU
        │
        ▼
   Dropout — 0.3
        │
        ▼
      Softmax
        │
        ▼
   Real / Fake
```

Global Average Pooling reduces spatial feature maps into a compact representation before classification.

Dropout is applied to reduce overfitting and improve generalization.

---

## Training Configuration

The model was implemented using **TensorFlow/Keras**.

| Configuration | Value |
|---|---|
| Architecture | Hybrid CNN |
| Backbone 1 | ResNet50 |
| Backbone 2 | DenseNet121 |
| Pretrained Weights | ImageNet |
| Input Shape | 256 × 256 × 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |
| Output Classes | 2 |
| Hidden Dense Layer | 256 |
| Dropout | 0.3 |
| Evaluation Metric | Accuracy |

---

## Training Strategy

Training deep neural networks effectively requires more than simply selecting an optimizer.

The training pipeline includes mechanisms for controlling convergence and reducing overfitting.

### ReduceLROnPlateau

The learning rate is automatically reduced when validation performance stops improving.

```python
ReduceLROnPlateau(
    factor=0.5,
    patience=2
)
```

This allows the optimizer to make progressively smaller parameter updates as training approaches convergence.

### Early Stopping

Early stopping prevents the network from continuing to train when validation loss no longer improves.

```python
EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

The best-performing weights are automatically restored.

---

## Model Performance

The hybrid architecture demonstrated strong performance across training, validation, and unseen test data.

### Training Performance

```text
Training Accuracy   ≈ 97.56%
Validation Accuracy ≈ 97.07%
Validation Loss     ≈ 0.0786
```

### Final Test Evaluation

The final model was evaluated on:

```text
20,000 unseen facial images
```

and achieved:

```text
Test Accuracy ≈ 96.94%
Test Loss     ≈ 0.0797
```

The validation and test results remain close, indicating that the learned representation generalized well to unseen samples rather than showing a large validation-to-test performance drop.

---

## OpenCV Face Localization

The classification network is combined with **OpenCV** to create an end-to-end facial analysis pipeline.

Face localization is performed using OpenCV's Haar Cascade detector:

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
```

The detector identifies facial bounding boxes before classification.

This separates the system into two stages:

```text
Stage 1
Face Localization
      │
      ▼
Stage 2
Deep Learning Classification
```

---

## End-to-End Inference Pipeline

During inference, the system performs the following sequence:

```text
Input Image
      │
      ▼
Detect Face
      │
      ▼
Extract Face ROI
      │
      ▼
Resize to 256 × 256
      │
      ▼
Normalize Pixels
      │
      ▼
Prepare Input Tensor
      │
      ▼
ResNet50 ────┐
             ├── Feature Fusion
DenseNet121 ─┘
      │
      ▼
Classification Head
      │
      ▼
Class Probabilities
      │
      ▼
Real / Fake Prediction
      │
      ▼
Display Bounding Box + Result
```

For each detected face, the system:

1. Detects the facial region.
2. Extracts the face ROI.
3. Resizes the ROI to the model input dimensions.
4. Normalizes the image.
5. Converts the image into the required tensor format.
6. Performs hybrid CNN inference.
7. Generates probabilities for both classes.
8. Selects the predicted class.
9. Displays the classification result.

---

## Prediction Output

The network generates probabilities for the two output classes.

An example output can be represented as:

```text
Fake Probability: 0.9861
Real Probability: 0.0139

Final Prediction: FAKE
```

or:

```text
Fake Probability: 0.0260
Real Probability: 0.9740

Final Prediction: REAL
```

The final prediction is derived from the model's output probabilities.

---

## Model Persistence

The trained architecture and learned weights can be saved separately.

```text
AAOHybrid_model.json
AAOHybrid_final.weights.h5
```

The architecture can later be reconstructed and the trained weights restored without retraining the complete network.

```python
from tensorflow.keras.models import model_from_json

with open("AAOHybrid_model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

model.load_weights(
    "AAOHybrid_final.weights.h5"
)
```

This allows the trained model to be reused for testing and inference.

---

## Technology Stack

### Deep Learning

- TensorFlow
- Keras
- ResNet50
- DenseNet121
- Convolutional Neural Networks
- Transfer Learning
- Feature Fusion

### Computer Vision

- OpenCV
- Haar Cascade
- Face Localization
- ROI Extraction
- Image Processing

### Machine Learning & Data Processing

- NumPy
- Pandas
- Scikit-learn
- ImageDataGenerator

### Visualization

- Matplotlib

### Programming Language

- Python

---

## Repository Structure

```text
Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121/
│
├── FINAL_NOTEBOOK.ipynb
│   └── Main experimentation, training and evaluation notebook
│
├── testing_notebook.ipynb
│   └── Model testing and inference experiments
│
├── training.py
│   └── Python training pipeline
│
└── README.md
    └── Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Habeba455/Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121.git
```

Navigate to the project:

```bash
cd Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121
```

Install the main dependencies:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn tqdm
```

---

## Running the Project

### Using Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
FINAL_NOTEBOOK.ipynb
```

### Using the Training Script

The training pipeline can also be executed through:

```bash
python training.py
```

Dataset paths must be configured according to the local environment before training.

---

## Engineering Highlights

This repository demonstrates more than a basic image classifier.

It covers multiple stages of a practical computer vision workflow:

```text
Data Preparation
      ↓
Transfer Learning
      ↓
Multi-Backbone CNN
      ↓
Feature Extraction
      ↓
Feature Fusion
      ↓
Regularized Classification
      ↓
Training Optimization
      ↓
Model Evaluation
      ↓
Face Localization
      ↓
Inference
      ↓
Model Persistence
```

### Key Technical Components

- Hybrid CNN architecture
- Dual pretrained CNN backbones
- ResNet50 feature extraction
- DenseNet121 feature extraction
- ImageNet transfer learning
- Deep feature concatenation
- Global Average Pooling
- Dense classification layers
- Dropout regularization
- Adaptive learning-rate reduction
- Early stopping
- Large-scale image training
- OpenCV face localization
- Face ROI preprocessing
- Binary Real/Fake classification
- Model serialization
- End-to-end inference workflow

---

## What Makes This Project Different?

A standard transfer-learning project typically relies on a single pretrained CNN followed by a classification layer.

This project instead explores a **multi-backbone feature-fusion strategy**:

```text
Single Backbone

Image → CNN → Classifier


Hybrid Approach

          ┌→ ResNet50 ────┐
Image ────┤               ├→ Feature Fusion → Classifier
          └→ DenseNet121 ─┘
```

The architecture allows the classifier to learn from complementary representations generated by two established deep CNN families.

Combined with face localization and preprocessing, this creates a complete pipeline from raw image input to final authenticity prediction.

---

## Future Improvements

Potential extensions include:

- Live webcam inference
- Video-stream processing
- MTCNN or RetinaFace for stronger face localization
- Precision, Recall and F1-score reporting
- ROC-AUC evaluation
- Project-specific confusion matrix generation
- Model quantization
- TensorFlow Lite deployment
- ONNX conversion
- GPU inference optimization
- Batch inference API
- FastAPI deployment
- Docker containerization
- Cloud deployment
- Temporal analysis for manipulated video detection

---

## Disclaimer

This project was developed for **research, educational, and computer vision experimentation purposes**.

Deep-learning predictions should not be considered definitive forensic evidence that an image is authentic or manipulated.

---

## Author

### Habeba Mohamed Fetouh

**AI / Machine Learning & Computer Vision Engineer**

Building practical AI systems across:

`Computer Vision` • `Deep Learning` • `Machine Learning` • `Python` • `TensorFlow` • `PyTorch` • `OpenCV`

---

<p align="center">
  <b>Built with Deep Learning, Computer Vision, and Feature-Level CNN Fusion.</b>
</p>

<p align="center">
  ⭐ If you find this project useful, consider starring the repository.
</p>
