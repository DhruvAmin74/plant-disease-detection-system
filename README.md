<div align="center">

<br/>

```
██████╗ ██╗      █████╗ ███╗   ██╗████████╗███╗   ███╗██████╗
██╔══██╗██║     ██╔══██╗████╗  ██║╚══██╔══╝████╗ ████║██╔══██╗
██████╔╝██║     ███████║██╔██╗ ██║   ██║   ██╔████╔██║██║  ██║
██╔═══╝ ██║     ██╔══██║██║╚██╗██║   ██║   ██║╚██╔╝██║██║  ██║
██║     ███████╗██║  ██║██║ ╚████║   ██║   ██║ ╚═╝ ██║██████╔╝
╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝╚═════╝
```

### *AI-Powered Plant Disease Detection*

<br/>

[![Live Demo](https://img.shields.io/badge/🌿_Live_Demo-PlantMD-4a7c2f?style=for-the-badge)](https://plant-disease-detection-system-gfbgq5o7nedrzdlx7msvxx.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-ff6f00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.13-d00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<br/>

> Upload a leaf photo. Get an instant diagnosis, description, and treatment plan — powered by EfficientNetV2-S trained on 87,000 images across 38 crop-disease classes.

<br/>

</div>

---

## 🌱 Overview

PlantMD is a production-grade deep learning web application for identifying plant diseases from leaf photographs. It leverages transfer learning on the **PlantVillage dataset** to classify 38 distinct crop-disease combinations with high accuracy, and pairs each prediction with a curated description and treatment recommendation.

The project covers the full ML pipeline — from raw data ingestion to cloud deployment — built as a reference implementation for agricultural computer vision.

---

## ✨ Features

- 🔍 **Instant diagnosis** — upload any leaf photo and get a result in seconds
- 🌾 **38 classes** — covers 14 crops including Tomato, Apple, Corn, Grape, Potato, and more
- 💊 **Treatment suggestions** — curated recommendations for every detected condition
- ⚠️ **Confidence indicator** — flags low-confidence predictions automatically
- 📱 **Fully responsive** — works on desktop, tablet, and mobile
- 🎨 **Botanical UI** — clean organic design built with Streamlit

---

## 🧠 Model Architecture

| Component | Detail |
|-----------|--------|
| **Base Model** | EfficientNetV2-S (ImageNet pretrained) |
| **Training Strategy** | Two-phase transfer learning |
| **Phase 1** | Frozen backbone — train classification head only |
| **Phase 2** | Unfreeze top 40% of backbone — fine-tune at 1e-5 LR |
| **Input Size** | 224 × 224 × 3 |
| **Output** | 38-class softmax |
| **Optimizer** | Adam + Cosine Decay scheduling |
| **Regularisation** | Dropout (0.4), BatchNorm, online augmentation, class weighting |
| **Precision** | Mixed precision (float16 compute / float32 variables) |
| **Parameters** | ~21.5M |

---

## 📊 Training Details

| Setting | Value |
|---------|-------|
| Dataset | PlantVillage (Kaggle) |
| Total Images | ~87,000 |
| Classes | 38 |
| Train / Val / Test Split | 70k / 8.8k / 8.8k |
| Batch Size | 64 |
| Phase 1 Epochs | 8 |
| Phase 2 Epochs | 5 |
| Hardware | NVIDIA T4 (Google Colab) |
| Framework | Keras 3.13 · TensorFlow 2.19 |

---

## 📁 Project Structure

```
plant-disease-detection/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application
│
├── model/
│   ├── plant_disease_efficientnet.keras   # Trained model weights
│   └── class_names.json                   # Class label mapping
│
├── notebook/
│   └── plant_disease_colab.ipynb          # Full training pipeline
│
├── assets/
│   ├── learning_curves.png       # Training history plot
│   ├── confusion_matrix.png      # Off-diagonal confusion matrix
│   ├── roc_auc.png               # ROC-AUC for hardest classes
│   └── gradcam.png               # Grad-CAM visualisations
│
├── outputs/
│   └── training_history.json     # Serialised training metrics
│
├── requirements.txt
├── runtime.txt                   # Python 3.11 for Streamlit Cloud
└── README.md
```

---

## 🚀 Run Locally

**1 — Clone the repo**
```bash
git clone https://github.com/DhruvAmin74/plant-disease-detection.git
cd plant-disease-detection
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**3 — Run the app**
```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`

---

## 🌿 Supported Crops & Conditions

| Crop | Conditions |
|------|-----------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌽 Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca, Leaf Blight, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🍊 Orange | Citrus Greening |
| 🫐 Blueberry | Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🌱 Soybean | Healthy |
| 🥦 Squash | Powdery Mildew |
| 🫐 Raspberry | Healthy |

---

## 🔬 Evaluation

The model is evaluated on a blind stratified test set (~8,800 images) carved from the validation corpus. Metrics include:

- **Macro F1-Score** — treats all 38 classes equally regardless of frequency
- **Weighted F1-Score** — accounts for class distribution
- **Off-diagonal Confusion Matrix** — correct predictions masked to surface misclassifications
- **ROC-AUC (One-vs-Rest)** — per-class AUC for the 5 hardest classes
- **Grad-CAM** — visual explanation of model attention on correct and incorrect predictions

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | EfficientNetV2-S via Keras 3 |
| Backend | TensorFlow 2.19 |
| Training | Google Colab (T4 GPU) |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Data | PlantVillage via Kaggle API |

---

## 📜 Dataset

**New Plant Diseases Dataset** — Kaggle  
Sourced from the original PlantVillage collection, augmented offline to ~87,000 images across 38 classes.  
License: copyright-authors  
[View on Kaggle →](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

<div align="center">

<br/>

*Built with 🌿 by [DhruvAmin74](https://github.com/DhruvAmin74)*

</div>
