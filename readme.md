# VGG16 Transfer Learning Image Classifier

This project demonstrates a deep learning image classifier built using **transfer learning** from the pre-trained **VGG16** convolutional neural network. It also includes **data augmentation** techniques to enhance model generalization and a **Streamlit web app** for easy model inference.

## 🚀 Features

- ✅ Transfer learning using **VGG16** from Keras Applications
- ✅ Image preprocessing and **data augmentation** via `ImageDataGenerator`
- ✅ Fine-tuning with custom fully-connected layers
- ✅ Model evaluation with accuracy and loss metrics
- ✅ Interactive **Streamlit app** to classify images in real-time

## 🧠 Model Architecture

- Base Model: **VGG16** (without top layers, weights from ImageNet)
- Custom Head: `Flatten` → `BatchNormalization` → `Dense` → `Sigmoid`
- Loss Function:`binary_crossentropy` *(depending on dataset)*
- Optimizer: `Adam`


