from pathlib import Path
import tensorflow as tf
import numpy as np

from .preprocessing import preprocess_image

CLASS_NAMES = [
    "AKIEC","BCC","BEN_OTH","BKL","DF",
    "INF","MAL_OTH","MEL","NV","SCCKA","VASC"
]

MODEL_PATH = Path(__file__).parent.parent / "models" / "multimodal_model.keras"

model = None


def load_model():
    global model

    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict(clinical_path, dermoscopic_path, metadata):

    model = load_model()

    clinical = preprocess_image(clinical_path)
    dermoscopic = preprocess_image(dermoscopic_path)

    clinical = np.expand_dims(clinical, 0)
    dermoscopic = np.expand_dims(dermoscopic, 0)
    metadata = np.expand_dims(metadata, 0)

    scores = model.predict(
        {
            "clinical": clinical,
            "dermoscopic": dermoscopic,
            "metadata": metadata
        },
        verbose=0
    )[0]

    prediction = CLASS_NAMES[np.argmax(scores)]

    return {
        "prediction": prediction,
        "confidence": float(np.max(scores)),
        "scores": dict(zip(CLASS_NAMES, scores.tolist()))
    }