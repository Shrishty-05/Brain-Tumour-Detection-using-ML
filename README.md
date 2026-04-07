
# 🧠 Brain Tumor Detection using Deep Learning

This project presents a deep learning-based system for **brain tumor detection** using MRI images. The system classifies MRI scans into **Tumor** and **No Tumor** categories. Multiple deep learning models including **CNN, MobileNetV2, and ResNet50** are trained and compared to determine the best-performing architecture. The final model is used to predict tumor presence from unseen MRI images.

---

# 🚀 Features

* Binary classification (Tumor / No Tumor)
* Multiple model comparison (CNN, MobileNetV2, ResNet50)
* Best model selection based on accuracy
* Confusion matrix visualization
* Accuracy comparison graph
* Training & validation curves
* Overfitting reduction using data augmentation
* Transfer learning using MobileNetV2

---

# 📁 Dataset

The dataset consists of brain MRI images categorized into two classes:

* Tumor
* No Tumor

Total Images: **1062**

* Tumor: 492
* No Tumor: 572

All images are resized to **224 × 224** and normalized before training.

---

# 🧠 Models Used

### 1. CNN (Custom Model)

A custom Convolutional Neural Network is built using convolutional, pooling, and dense layers. The model learns spatial features directly from MRI images.

### 2. MobileNetV2

Transfer learning is used with MobileNetV2 pretrained on ImageNet. Custom classification layers are added and fine-tuned for tumor classification.

### 3. ResNet50

ResNet50 architecture is used with transfer learning. The pretrained layers extract deep features for classification.

---

# ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn

---

# 📊 Training Process

1. Load MRI dataset
2. Preprocess images (resize, normalize)
3. Train-test split
4. Train CNN, MobileNetV2, ResNet50
5. Evaluate models
6. Compare accuracy
7. Save best model

---

# 📈 Model Performance

| Model       | Accuracy |
| ----------- | -------- |
| CNN         | ~90%     |
| MobileNetV2 | ~82%  |
| ResNet50    | ~63%  |

MobileNetV2 achieved the best performance.

---

# 🧪 Prediction

The trained model predicts:

* 🧠 Tumor Detected (YES)
* ✅ No Tumor (NO)

---


# 📂 Project Structure

```
brain-tumor-detection
│
├── data/
│   ├── yes
│   └── no
│
├── train.ipynb
├── best_brain_tumor_model.h5
└── README.md
```

---

# 🎯 Applications

* Medical image analysis
* Brain tumor screening
* AI-assisted diagnosis
* Research and academic projects

---

# 📌 Future Work

* Multi-class tumor classification
* Localization of tumor region
* Web deployment
* Larger medical dataset training
* Clinical testing

