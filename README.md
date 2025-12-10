# Image Caption Generator

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
```

## 📊 Evaluation and Results

Model evaluation uses **BLEU scores** to measure how closely generated captions match reference captions.

| Metric  | Score |
|---------|-------|
| BLEU-1  | 0.54  |

A BLEU score of **0.54** (for one-grams) reflects strong alignment with human-generated captions.  
Visual inspection confirms the captions are coherent, descriptive, and relevant to the input images.

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **VGG16** (pre-trained on ImageNet)

---

## 🏁 Conclusion

This project successfully demonstrates how **CNNs** and **LSTMs** can be combined to interpret and describe visual content.  
It showcases the potential of deep learning in creating intelligent systems that understand and generate human-like language.

