# Brain-Tumor-detection_202401100300185

# 🧠 Brain Tumor Detection Using CNN

A deep learning-based project for classifying brain MRI images as **Tumor** or **No Tumor** using Convolutional Neural Networks (CNNs). This project is designed to aid in early diagnosis by providing an automated, efficient, and accurate classification model.

---

## 🚀 Project Overview:-

Brain tumors are one of the most serious and life-threatening health conditions. Early detection through medical imaging can significantly improve treatment outcomes. This project:

- Uses a **custom CNN model** to classify MRI images.
- Provides **visual feedback** for predictions.
- Evaluates model performance using **accuracy, precision, recall, F1-score**, and **confusion matrix**.

---

## 🎥 Demo:-

Below is an example of a model prediction:

| MRI Image | Predicted Label |
|-----------|-----------------|
| ![Example](outputs/sample_prediction.png) | Tumor |

---

## 📂 Dataset:-

The dataset used for training and testing is sourced from Kaggle:

- 📁 [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Two classes:  
  - `yes/`: MRI images with tumors  
  - `no/`: MRI images without tumors

---

🧠 Model Architecture:- 

The CNN model consists of:
Multiple Conv2D layers with ReLU activation
MaxPooling2D for down-sampling
Dropout layers for regularization
Fully connected Dense layers
Final layer with Sigmoid activation for binary classification

---

📊 Results & Evaluation:- 

Accuracy: ~93% on validation set
Metrics Used:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix

---

🙏 Acknowledgments:- 

Kaggle MRI Dataset by Navoneel Chakrabarty
TensorFlow
Keras
scikit-learn
Matplotlib
NumPy
