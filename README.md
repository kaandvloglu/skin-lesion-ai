# Skin Lesion AI

A multimodal AI-powered clinical decision support system for skin lesion classification using the **MILK10K** dataset.

The project combines **clinical images**, **dermoscopic images**, and **patient metadata** to classify skin lesions into **11 diagnostic categories**, while also providing **Grad-CAM visual explanations** to improve prediction interpretability.

---

## Project Overview

This project was developed as a Software Engineering Capstone Project and consists of three integrated components:

* **AI Module:** Multimodal deep learning model built with EfficientNetB3.
* **Backend:** REST API that connects the AI model with the frontend.
* **Frontend:** User interface for image upload, prediction visualization, and Grad-CAM display.

The system analyzes paired skin lesion images together with patient metadata to produce accurate and explainable predictions.

---

## Features

* Multimodal image classification
* Clinical + dermoscopic image support
* Patient metadata integration
* Top-3 predictions with confidence scores
* Grad-CAM visual explanations
* Backend API integration
* Frontend visualization interface

---

## Model Architecture

The AI model is based on **EfficientNetB3** with **transfer learning** from ImageNet.

### Training Strategy

* Transfer learning using EfficientNetB3
* Fine-tuning of upper backbone layers
* Data augmentation (flip, rotation, zoom, contrast)
* Adam optimizer
* Cross-entropy loss
* 25 training epochs

---

## Inference Pipeline

1. User uploads clinical and dermoscopic images.
2. Images are resized to **300×300**.
3. Patient metadata is one-hot encoded.
4. Features are fused inside the multimodal model.
5. Softmax generates probabilities for **11 classes**.
6. Grad-CAM heatmaps are generated for both image modalities.

---

## Final Model Performance

| Metric               | Value     |
| -------------------- | --------- |
| Training Samples     | 4,192     |
| Validation Samples   | 1,048     |
| Total Paired Samples | 5,240     |
| Validation Accuracy  | **54.2%** |
| Validation Loss      | **1.44**  |
| Weighted F1          | **41.3%** |
| Macro F1             | **11.5%** |
| Epochs               | **25**    |

---

## Supported Classes

The model predicts the following lesion categories:

* AKIEC
* BCC
* BEN_OTH
* BKL
* DF
* INF
* MAL_OTH
* MEL
* NV
* SCCKA
* VASC

---

## Example Prediction

```text
Prediction
------------------------------
1. BCC   51.09%
2. BKL   12.58%
3. MEL   10.14%
```

Example API response:

```json
{
  "prediction": "BCC",
  "confidence": 0.5109,
  "top3": [
    { "label": "BCC", "confidence": 0.5109 },
    { "label": "BKL", "confidence": 0.1258 },
    { "label": "MEL", "confidence": 0.1014 }
  ]
}
```

---

## Explainability

To improve transparency, the system generates **Grad-CAM** heatmaps for both uploaded images.

The frontend displays:

* Clinical image heatmap
* Dermoscopic image heatmap
* Prediction confidence scores

This helps visualize which regions influenced the model's decision.

---

## Project Structure

```text
skin-lesion-ai/
├── ai/
│   ├── models/
│   ├── src/
│   ├── data/
│   └── requirements.txt
├── backend/
├── frontend/
├── README.md
└── LICENSE
```

---

## Tech Stack

### AI

* Python
* TensorFlow / Keras
* EfficientNetB3
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn

### Backend

* Spring Boot
* REST API

### Frontend

* Streamlit

---

## Dataset

The project is trained using the **MILK10K** skin lesion dataset, which contains paired clinical and dermoscopic images together with patient metadata.

The metadata includes:

* Age
* Sex
* Skin tone
* Anatomical site

---

## Installation

### AI Module

```bash
cd ai
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
streamlit run app.py
```

### Backend

Run the Spring Boot backend to expose the API endpoints used by the frontend.

---

## Project Status

| Feature                   | Status |
| ------------------------- | ------ |
| Multimodal model training | ✅      |
| Validation pipeline       | ✅      |
| Inference pipeline        | ✅      |
| Metadata encoding         | ✅      |
| Top-3 predictions         | ✅      |
| Grad-CAM visualization    | ✅      |
| Backend integration       | ✅      |
| Frontend integration      | ✅      |
| End-to-end system         | ✅      |

---

## Team

This project was developed as a Software Engineering Capstone Project.

* **Kaan Develioğlu** — AI Model Development
* **Yaman** — Backend Development
* **Eren** — Frontend Development