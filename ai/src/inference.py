from pathlib import Path
import tensorflow as tf
import numpy as np

from .preprocessing import preprocess_image

CLASS_NAMES = [
    "AKIEC","BCC","BEN_OTH","BKL","DF",
    "INF","MAL_OTH","MEL","NV","SCCKA","VASC"
]

MODEL_PATH = Path(__file__).parent.parent / "models" / "multimodal_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict(clinical_path, dermoscopic_path, metadata):

    clinical = preprocess_image(clinical_path)
    derm = preprocess_image(dermoscopic_path)

    clinical = np.expand_dims(clinical, 0)
    derm = np.expand_dims(derm, 0)
    meta = np.expand_dims(metadata, 0)

    scores = model.predict(
        {
            "clinical": clinical,
            "dermoscopic": derm,
            "metadata": meta
        },
        verbose=0
    )[0]

    return {
        "prediction": CLASS_NAMES[np.argmax(scores)],
        "confidence": float(np.max(scores)),
        "scores": dict(zip(CLASS_NAMES, scores.tolist()))
    }