from pathlib import Path
import tensorflow as tf
import numpy as np

from .preprocessing import preprocess_image

CLASS_NAMES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

MODEL_PATH = Path(__file__).parent.parent / "models" / "multimodal_model.keras"

print("DEBUG: Loading AI model...", flush=True)

model = tf.keras.models.load_model(MODEL_PATH)

print("DEBUG: AI model loaded successfully.", flush=True)


def predict(clinical_path, dermoscopic_path, metadata):

    print("DEBUG 1: predict() started", flush=True)

    print("DEBUG 2: preprocessing clinical image...", flush=True)
    clinical = preprocess_image(clinical_path)
    print("DEBUG 3: clinical image ready", flush=True)

    print("DEBUG 4: preprocessing dermoscopic image...", flush=True)
    derm = preprocess_image(dermoscopic_path)
    print("DEBUG 5: dermoscopic image ready", flush=True)

    clinical = np.expand_dims(clinical, 0)
    derm = np.expand_dims(derm, 0)
    meta = np.expand_dims(metadata, 0)

    print("DEBUG 6: model inputs prepared", flush=True)
    print(
        f"DEBUG: clinical={clinical.shape}, "
        f"dermoscopic={derm.shape}, "
        f"metadata={meta.shape}",
        flush=True
    )

    print("DEBUG 7: model.predict() starting...", flush=True)

    scores = model.predict(
        {
            "clinical": clinical,
            "dermoscopic": derm,
            "metadata": meta
        },
        verbose=0
    )[0]

    print("DEBUG 8: model.predict() completed", flush=True)

    result = {
        "prediction": CLASS_NAMES[np.argmax(scores)],
        "confidence": float(np.max(scores)),
        "scores": dict(zip(CLASS_NAMES, scores.tolist()))
    }

    print(
        f"DEBUG 9: prediction result ready: {result['prediction']}",
        flush=True
    )

    return result