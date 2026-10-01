# 🧠 Real-Time Face Detection using Hybrid CNN
## ResNet50 + DenseNet121 Deep Learning Architecture

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv">
  <img src="https://img.shields.io/badge/Model-ResNet50%20%2B%20DenseNet121-purple">
  <img src="https://img.shields.io/badge/Task-Real%20vs%20Fake-red">
</p>

A deep-learning-based computer vision system for **face detection and Real/Fake face classification** using a hybrid CNN architecture combining **ResNet50** and **DenseNet121**.

The system integrates transfer learning, deep feature extraction, feature fusion, face localization, and classification into a complete computer vision pipeline.

The model was trained and evaluated on a large-scale dataset containing **140,000 facial images** and achieved approximately **96.94% test accuracy**.

---

# 📌 Project Overview

Distinguishing real facial images from artificially generated or manipulated faces is a challenging computer vision problem.

Traditional single-backbone CNN architectures may capture only a limited representation of complex facial patterns.

This project introduces a **Hybrid CNN Architecture** that combines two powerful pretrained convolutional neural networks:

- **ResNet50**
- **DenseNet121**

Both networks independently extract high-level visual features from the same facial image.

Their feature representations are then combined through **feature-level fusion**, producing a richer representation before final classification.

The complete pipeline performs:

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
Image Preprocessing
     │
     ▼
ResNet50 + DenseNet121
     │
     ▼
Deep Feature Fusion
     │
     ▼
Classification
     │
     ▼
REAL / FAKE
```

---

# 🏗️ Hybrid CNN Architecture

The core of the system is a dual-backbone deep learning architecture.

The same facial image is processed simultaneously by **ResNet50** and **DenseNet121**.

```text
                    Input Face
                   256 × 256 × 3
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          ResNet50              DenseNet121
       ImageNet Weights       ImageNet Weights
              │                     │
              ▼                     ▼
       Deep Features          Deep Features
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                Feature Concatenation
                         │
                         ▼
               Global Average Pooling
                         │
                         ▼
                  Dense Layer
                    256 Units
                         │
                         ▼
                       ReLU
                         │
                         ▼
                    Dropout
                         │
                         ▼
                     Softmax
                         │
                  ┌──────┴──────┐
                  ▼             ▼
                REAL           FAKE
```

### Why ResNet50?

ResNet50 introduces **residual connections**, allowing deep neural networks to learn complex representations while reducing problems associated with vanishing gradients.

It provides strong hierarchical feature extraction for visual recognition tasks.

### Why DenseNet121?

DenseNet121 uses **dense connectivity**, where layers receive information from previous layers.

This encourages:

- Feature reuse
- Efficient gradient propagation
- Preservation of low-level information
- Strong visual representation learning

### Why Combine Them?

Instead of depending on a single CNN backbone, the hybrid model learns complementary representations from both architectures.

ResNet50 contributes strong residual representations, while DenseNet121 provides densely connected feature propagation.

The resulting features are fused before classification.

---

# 🔬 Feature Extraction & Fusion

The hybrid approach is based on extracting deep representations from multiple CNN branches and combining them into a unified feature space.

<p align="center">
  <img src="images/hybrid_architecture.png" width="900">
</p>

<p align="center">
  <em>Illustration of a multi-backbone feature extraction and feature-fusion architecture.</em>
</p>

> **Note:** The figure above illustrates the general concept of multi-backbone feature extraction. The architecture implemented in this repository specifically uses **ResNet50 + DenseNet121**.

The implemented feature pipeline can be summarized as:

```text
ResNet50 Features
       │
       ├──────────────┐
       │              │
       │        Feature Fusion
       │              │
DenseNet121 Features ─┘
                      │
                      ▼
              Combined Feature Map
                      │
                      ▼
            Global Average Pooling
                      │
                      ▼
             Fully Connected Layer
                      │
                      ▼
               Classification
```

---

# 📊 Dataset

The model was trained using a large-scale **Real and Fake Faces dataset**.

The complete dataset contains:

| Dataset Split | Number of Images |
|---|---:|
| Training | **100,000** |
| Validation | **20,000** |
| Testing | **20,000** |
| **Total** | **140,000** |

The dataset contains two classes:

```text
Real
Fake
```

Dataset structure:

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

# ⚙️ Image Preprocessing

Before entering the neural network, images are prepared for inference.

The preprocessing pipeline includes:

```text
Input Image
      │
      ▼
Face Localization
      │
      ▼
Face ROI Extraction
      │
      ▼
Resize
256 × 256
      │
      ▼
Pixel Normalization
      │
      ▼
Batch Dimension
      │
      ▼
Hybrid CNN
```

Images are resized to:

```text
256 × 256 × 3
```

Pixel values are normalized before being passed to the neural network.

---

# 🧠 Training Configuration

The hybrid network was implemented using **TensorFlow/Keras**.

| Parameter | Configuration |
|---|---|
| Backbone 1 | ResNet50 |
| Backbone 2 | DenseNet121 |
| Pretrained Weights | ImageNet |
| Input Resolution | 256 × 256 × 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Output Classes | 2 |
| Classification | Real / Fake |

---

# 🎯 Transfer Learning

Both CNN backbones use pretrained **ImageNet weights**.

Transfer learning allows the system to start from visual representations learned from millions of images rather than training every convolutional feature from scratch.

This provides useful representations for:

- Edges
- Shapes
- Textures
- Facial structures
- High-level visual patterns

These representations are then adapted to the Real/Fake classification task.

---

# 🛡️ Regularization

Dropout is incorporated into the network to reduce overfitting and improve generalization.

```python
Dropout(0.3)
```

Regularization becomes particularly important when combining large pretrained CNN architectures.

---

# 📉 Adaptive Learning Rate

The training pipeline uses **ReduceLROnPlateau**.

```python
ReduceLROnPlateau(
    factor=0.5,
    patience=2
)
```

When validation performance stops improving, the learning rate is automatically reduced.

This allows the optimizer to perform smaller parameter updates during later stages of training.

---

# ⏹️ Early Stopping

Early stopping is used to prevent unnecessary training after validation performance stops improving.

```python
EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

The best-performing model weights are automatically restored.

---

# 📈 Model Performance

The hybrid architecture achieved strong performance on both validation and unseen test data.

| Metric | Result |
|---|---:|
| Training Accuracy | **~97.56%** |
| Validation Accuracy | **~97.07%** |
| Test Accuracy | **~96.94%** |
| Validation Loss | **~0.0786** |
| Test Loss | **~0.0797** |

### Test Set

The final model was evaluated on:

```text
20,000 unseen images
```

Final evaluation:

```text
Test Accuracy ≈ 96.94%
Test Loss     ≈ 0.0797
```

The small difference between validation and test performance suggests strong generalization to unseen samples.

---

# 📊 Confusion Matrix Analysis

Confusion matrices provide a deeper view of classification performance than accuracy alone.

They show the relationship between:

- True Positives
- True Negatives
- False Positives
- False Negatives

and help identify systematic classification errors.

<p align="center">
  <img src="images/confusion_matrix.png" width="850">
</p>

<p align="center">
  <em>Example confusion-matrix visualization for evaluating classification performance.</em>
</p>

> **Important:** The figure above is included as a reference visualization. It should not be interpreted as the Real/Fake confusion matrix produced by this repository unless it was generated directly from this project's predictions.

For this project, the final test evaluation reached approximately:

```text
Accuracy = 96.94%
Loss     = 0.0797
```

---

# 👁️ Face Detection Pipeline

The deep-learning classifier is integrated with **OpenCV** for face localization.

OpenCV's Haar Cascade classifier is used to identify facial regions.

```python
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
```

The inference workflow is:

```text
Input Image
     │
     ▼
OpenCV Face Detector
     │
     ▼
Bounding Box
     │
     ▼
Face ROI
     │
     ▼
Resize to 256 × 256
     │
     ▼
Normalize
     │
     ▼
Hybrid CNN
     │
     ▼
Class Probabilities
     │
     ▼
Real / Fake Prediction
```

---

# 🔍 Inference Pipeline

For every detected face, the system performs the following operations:

1. Detect the face using OpenCV.
2. Generate the facial bounding box.
3. Extract the facial Region of Interest.
4. Resize the ROI to `256 × 256`.
5. Normalize the image.
6. Convert it into the required tensor format.
7. Pass it through the hybrid CNN.
8. Generate probabilities for Real and Fake classes.
9. Select the predicted class.
10. Display the classification result.

---

# 🧪 Prediction

The model outputs probabilities for two classes:

```text
Class 0 → Fake
Class 1 → Real
```

Example prediction:

```text
Fake Probability: 0.9861
Real Probability: 0.0139

Prediction → FAKE
```

Another example:

```text
Fake Probability: 0.0260
Real Probability: 0.9740

Prediction → REAL
```

---

# 💾 Model Persistence

The trained architecture and learned weights can be stored separately.

```text
AAOHybrid_model.json
AAOHybrid_final.weights.h5
```

The model architecture can then be reconstructed:

```python
from tensorflow.keras.models import model_from_json

with open("AAOHybrid_model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

model.load_weights(
    "AAOHybrid_final.weights.h5"
)
```

This allows inference without retraining the network.

---

# 🛠️ Technology Stack

### Programming

![Python](https://img.shields.io/badge/Python-Programming-blue?logo=python)

### Deep Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-orange?logo=tensorflow)

![Keras](https://img.shields.io/badge/Keras-Neural_Networks-red?logo=keras)

### Computer Vision

![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?logo=opencv)

### Core Architectures

```text
ResNet50
DenseNet121
Hybrid CNN
Transfer Learning
Feature Fusion
```

### Data & Analysis

```text
NumPy
Pandas
Scikit-learn
Matplotlib
```

---

# 📁 Repository Structure

```text
Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121/
│
├── FINAL_NOTEBOOK.ipynb
│
├── testing_notebook.ipynb
│
├── training.py
│
├── README.md
│
├── images/
│   ├── hybrid_architecture.png
│   └── confusion_matrix.png
│
└── models/
    ├── AAOHybrid_model.json
    └── AAOHybrid_final.weights.h5
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Habeba455/Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121.git
```

Navigate to the project directory:

```bash
cd Deep-Learning-Real-Time-Face-Detection-ResNet50-DenseNet121
```

Install dependencies:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn tqdm
```

---

# ▶️ Running the Project

### Jupyter Notebook

Launch:

```bash
jupyter notebook
```

Then open:

```text
FINAL_NOTEBOOK.ipynb
```

### Training Script

The training pipeline can also be executed through:

```bash
python training.py
```

Dataset paths should be configured according to the local environment before running the training pipeline.

---

# 💡 Core Technical Contributions

This project demonstrates an end-to-end computer vision workflow involving:

- Hybrid CNN architecture design
- ResNet50 feature extraction
- DenseNet121 feature extraction
- Transfer learning with ImageNet weights
- Multi-backbone feature fusion
- Deep feature concatenation
- Global Average Pooling
- Fully connected classification
- Dropout regularization
- Adaptive learning-rate scheduling
- Early stopping
- Large-scale image training
- Model evaluation
- Model serialization
- OpenCV face detection
- Face ROI extraction
- Image preprocessing
- Real/Fake face classification
- End-to-end inference

---

# 🔮 Future Improvements

Future development could extend the current system with:

- Real-time webcam streaming
- Video-based inference
- RetinaFace or MTCNN face detection
- Precision and Recall evaluation
- F1-score analysis
- ROC-AUC evaluation
- Project-specific confusion matrix visualization
- TensorFlow Lite deployment
- ONNX conversion
- GPU inference optimization
- FastAPI inference API
- Docker containerization
- Cloud deployment
- Temporal deepfake detection

---

# ⚠️ Disclaimer

This project was developed for **educational, research, and computer vision experimentation purposes**.

Deep-learning predictions should not be considered definitive forensic evidence that an image is authentic or manipulated.

---

# 👩‍💻 Author

## Habeba Mohamed Fetouh

**AI / Machine Learning & Computer Vision Engineer**

Focused on building practical AI systems using:

`Python` • `Deep Learning` • `Computer Vision` • `TensorFlow` • `PyTorch` • `OpenCV` • `Machine Learning`

---

<p align="center">
  <b>⭐ If you find this project useful, consider starring the repository.</b>
</p>
