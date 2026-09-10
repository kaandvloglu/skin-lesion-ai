from pathlib import Path
import tensorflow as tf
import numpy as np

from .preprocessing import preprocess_image


CLASS_NAMES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
    "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]


MODEL_PATH = Path(__file__).parent.parent / "models" / "multimodal_model.tflite"

interpreter = None
input_details = None
output_details = None


def get_interpreter():
    global interpreter, input_details, output_details

    if interpreter is None:
        print("DEBUG: Loading TFLite model...", flush=True)

        interpreter = tf.lite.Interpreter(
            model_path=str(MODEL_PATH)
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("DEBUG: TFLite model loaded successfully.", flush=True)

    return interpreter


def predict(clinical_path, dermoscopic_path, metadata):

    print("DEBUG 1: predict() started", flush=True)

    print("DEBUG 2: preprocessing clinical image...", flush=True)
    clinical = preprocess_image(clinical_path)
    print("DEBUG 3: clinical image ready", flush=True)

    print("DEBUG 4: preprocessing dermoscopic image...", flush=True)
    derm = preprocess_image(dermoscopic_path)
    print("DEBUG 5: dermoscopic image ready", flush=True)

    clinical = np.expand_dims(clinical, 0).astype(np.float32)
    derm = np.expand_dims(derm, 0).astype(np.float32)
    meta = np.expand_dims(metadata, 0).astype(np.float32)

    print("DEBUG 6: TFLite inputs prepared", flush=True)

    print(
        f"DEBUG: clinical={clinical.shape}, "
        f"dermoscopic={derm.shape}, "
        f"metadata={meta.shape}",
        flush=True
    )

    current_interpreter = get_interpreter()

    print("DEBUG 7: TFLite inference starting...", flush=True)

    current_interpreter.set_tensor(
        input_details[0]["index"],
        clinical
    )

    current_interpreter.set_tensor(
        input_details[1]["index"],
        derm
    )

    current_interpreter.set_tensor(
        input_details[2]["index"],
        meta
    )

    current_interpreter.invoke()

    scores = current_interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    print("DEBUG 8: TFLite inference completed", flush=True)

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