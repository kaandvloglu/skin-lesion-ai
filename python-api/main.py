from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import tempfile
import json
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from ai.src.inference import predict
from ai.src.preprocessing import encode_metadata, preprocess_image
from ai.src.gradcam import make_gradcam_heatmap, heatmap_to_base64


ROOT = Path(__file__).resolve().parent.parent

KERAS_MODEL_PATH = ROOT / "ai" / "models" / "multimodal_model.keras"

with open(ROOT / "ai" / "models" / "metadata_columns.json") as f:
    TRAINING_COLUMNS = json.load(f)


app = FastAPI(title="Skin Lesion AI API")

# Keras model is loaded only when Grad-CAM is requested.
gradcam_model = None


def get_gradcam_model():
    global gradcam_model

    if gradcam_model is None:
        print("DEBUG: Loading Keras model for Grad-CAM...", flush=True)

        gradcam_model = tf.keras.models.load_model(
            KERAS_MODEL_PATH,
            compile=False
        )

        print("DEBUG: Keras Grad-CAM model loaded.", flush=True)

    return gradcam_model


@app.get("/")
def root():
    return {"message": "AI service is running"}


@app.get("/model-info")
def model_info():
    from ai.src.inference import get_interpreter

    current_interpreter = get_interpreter()

    inputs = current_interpreter.get_input_details()
    outputs = current_interpreter.get_output_details()

    return {
        "model_loaded": True,
        "model_type": "TFLite",
        "input_shapes": [
            detail["shape"].tolist()
            for detail in inputs
        ],
        "output_shapes": [
            detail["shape"].tolist()
            for detail in outputs
        ]
    }


@app.post("/predict")
async def predict_endpoint(
    clinical_image: UploadFile = File(...),
    dermoscopic_image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    skin_tone: int = Form(...),
    site: str = Form(...)
):
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as c_file:
        c_file.write(await clinical_image.read())
        clinical_path = Path(c_file.name)

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as d_file:
        d_file.write(await dermoscopic_image.read())
        dermoscopic_path = Path(d_file.name)

    try:
        df = pd.DataFrame([{
            "age_approx": age,
            "sex": sex,
            "skin_tone_class": skin_tone,
            "site": site
        }])

        metadata = encode_metadata(
            df,
            columns=TRAINING_COLUMNS
        ).iloc[0].to_numpy(dtype=np.float32)

        result = predict(
            clinical_path,
            dermoscopic_path,
            metadata
        )

        return result

    finally:
        clinical_path.unlink(missing_ok=True)
        dermoscopic_path.unlink(missing_ok=True)


@app.post("/gradcam")
async def gradcam_endpoint(
    clinical_image: UploadFile = File(...),
    dermoscopic_image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    skin_tone: int = Form(...),
    site: str = Form(...)
):
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as c_file:
        c_file.write(await clinical_image.read())
        clinical_path = Path(c_file.name)

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as d_file:
        d_file.write(await dermoscopic_image.read())
        dermoscopic_path = Path(d_file.name)

    try:
        print("DEBUG: Grad-CAM request started", flush=True)

        df = pd.DataFrame([{
            "age_approx": age,
            "sex": sex,
            "skin_tone_class": skin_tone,
            "site": site
        }])

        metadata = encode_metadata(
            df,
            columns=TRAINING_COLUMNS
        ).iloc[0].to_numpy(dtype=np.float32)

        clinical = preprocess_image(clinical_path)
        dermoscopic = preprocess_image(dermoscopic_path)

        clinical = tf.expand_dims(clinical, 0)
        dermoscopic = tf.expand_dims(dermoscopic, 0)
        metadata_tensor = tf.expand_dims(metadata, 0)

        inputs = {
            "clinical": clinical,
            "dermoscopic": dermoscopic,
            "metadata": metadata_tensor
        }

        model = get_gradcam_model()

        # -----------------------------
        # CLINICAL GRAD-CAM
        # -----------------------------

        print(
            "DEBUG: Creating clinical Grad-CAM...",
            flush=True
        )

        clinical_heatmap = make_gradcam_heatmap(
            model,
            inputs,
            branch="clinical"
        )

        clinical_gradcam = heatmap_to_base64(
            clinical,
            clinical_heatmap
        )

        # Free heatmap memory before starting the second Grad-CAM.
        del clinical_heatmap
        gc.collect()

        print(
            "DEBUG: Clinical Grad-CAM completed.",
            flush=True
        )

        # -----------------------------
        # DERMOSCOPIC GRAD-CAM
        # -----------------------------

        print(
            "DEBUG: Creating dermoscopic Grad-CAM...",
            flush=True
        )

        dermoscopic_heatmap = make_gradcam_heatmap(
            model,
            inputs,
            branch="dermoscopic"
        )

        dermoscopic_gradcam = heatmap_to_base64(
            dermoscopic,
            dermoscopic_heatmap
        )

        del dermoscopic_heatmap
        gc.collect()

        print(
            "DEBUG: Dermoscopic Grad-CAM completed.",
            flush=True
        )

        print(
            "DEBUG: Grad-CAM completed",
            flush=True
        )

        return {
            "clinical_gradcam": clinical_gradcam,
            "dermoscopic_gradcam": dermoscopic_gradcam
        }

    finally:
        clinical_path.unlink(missing_ok=True)
        dermoscopic_path.unlink(missing_ok=True)

        gc.collect()