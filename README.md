# skin-lesion-ai
Multimodal Clinical Decision Support System for Skin Lesion Classification using MILK10k Dataset

## Final Model Performance

The final multimodal model combines clinical images, dermoscopic images, and patient metadata using an EfficientNetB3 backbone.

### Results

| Metric              | Value     |
| ------------------- | --------- |
| Training Samples    | 4,192     |
| Validation Samples  | 1,048     |
| Paired Samples      | 5,240     |
| Validation Accuracy | **54.9%** |
| Validation Loss     | **1.44**  |
| Epochs              | 25        |

### Example Prediction

```
Prediction
------------------------------
1. BCC      51.09%
2. BKL      12.58%
3. MEL      10.14%
```

The inference pipeline successfully predicts the three most probable lesion classes using both image modalities and encoded metadata.

## Final Model Performance

- Validation Accuracy:
- Macro F1:
- Weighted F1:
- Epochs: 25
- Framework: TensorFlow/Keras