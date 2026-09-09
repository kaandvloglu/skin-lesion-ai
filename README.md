# Skin Lesion AI

A multimodal clinical decision support system for skin lesion classification using the **MILK10K** dataset. The model combines **clinical images**, **dermoscopic images**, and **patient metadata** to classify skin lesions into 11 diagnostic categories.

## Project Overview

This project uses a multimodal deep learning architecture based on **EfficientNetB3** to improve skin lesion classification by leveraging multiple sources of information:

- Clinical images
- Dermoscopic images
- Patient metadata (age, sex, skin tone, anatomical site)

The inference pipeline returns the predicted class, confidence scores, Top-3 predictions, and Grad-CAM visual explanations for both image modalities.

## Final Model Performance

| Metric | Value |
|--------|-------|
| Training Samples | 4,192 |
| Validation Samples | 1,048 |
| Paired Samples | 5,240 |
| Validation Accuracy | **54.2%** |
| Validation Loss | **1.44** |
| Macro F1 | **11.5%** |
| Weighted F1 | **41.3%** |
| Epochs | **25** |
| Framework | TensorFlow / Keras |

## Supported Classes

The model predicts the following 11 lesion categories:

- AKIEC
- BCC
- BEN_OTH
- BKL
- DF
- INF
- MAL_OTH
- MEL
- NV
- SCCKA
- VASC

## Example Prediction

```text
Prediction
------------------------------
1. BCC      51.09%
2. BKL      12.58%
3. MEL      10.14%
```

Example JSON output:

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

## Explainability

The inference pipeline also generates **Grad-CAM** visual explanations for both image modalities.

- Clinical image Grad-CAM
- Dermoscopic image Grad-CAM

The heatmaps are returned as Base64-encoded PNG images, allowing the frontend to display visual explanations directly.

## Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB3
- Pandas
- NumPy
- OpenCV
- Matplotlib

## Dataset

The project is trained using the **MILK10K** skin lesion dataset, containing paired clinical and dermoscopic images with accompanying patient metadata.

## Project Status

- ✅ Multimodal model training completed
- ✅ Validation pipeline completed
- ✅ Inference pipeline completed
- ✅ Top-3 prediction support
- ✅ Grad-CAM visualization
- ✅ Metadata encoding
- ✅ Ready for backend integration