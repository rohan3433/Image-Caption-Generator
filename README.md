# 🖼️ Image Caption Generator

An innovative deep learning project that combines **Computer Vision** and **Natural Language Processing (NLP)** to automatically generate descriptive captions for images.

---

## 📘 Introduction

This project implements an **Image Caption Generator** using the **Flickr dataset**, leveraging **Keras** and **TensorFlow** as the core frameworks.  
The goal is to develop a model that can analyze images and produce meaningful captions that accurately describe visual content.

---

## 🧩 Project Overview

The model works by:
1. Extracting image features using **Convolutional Neural Networks (CNNs)** with the **VGG16** architecture.
2. Using **Long Short-Term Memory (LSTM)** networks to process textual data and generate captions.
3. Combining visual and textual representations to predict accurate word sequences describing the image.

---

## 🧹 Data Preparation

1. **Image Preprocessing**
   - Resize all images to **224×224 pixels**.
   - Convert images into **NumPy arrays**.
   - Apply VGG16’s preprocessing function for normalization.

2. **Text Preprocessing**
   - Convert all captions to lowercase.
   - Remove punctuation and special characters.
   - Tokenize captions and create word-index mappings.
   - Map each image ID to its corresponding cleaned captions.

---

## 🏗️ Model Architecture

The architecture integrates visual and textual components:
- **CNN Encoder:** Extracts feature vectors from VGG16’s final layer.
- **LSTM Decoder:** Learns the sequence of words describing the image.
- **Dense Layers:** Combine both modalities and predict the next word using a **Softmax** activation.

Layers used:
- Dense layers  
- LSTM layers  
- Dropout layers (to prevent overfitting)

---

## 🏋️ Training the Model

- The model is trained for multiple epochs with a defined **batch size**.
- A custom **data generator** feeds batches of image features and corresponding caption sequences.
- **Loss and accuracy metrics** are monitored during training to evaluate learning progress.

Example:
```python
model.fit(generator, epochs=15, steps_per_epoch=steps, verbose=1)
model.save('best_model.h5')
