from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import tempfile
import json
import numpy as np
import pandas as pd

from ai.src.inference import predict
from ai.src.preprocessing import encode_metadata


ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "ai" / "models" / "metadata_columns.json") as f:
    TRAINING_COLUMNS = json.load(f)


app = FastAPI(title="Skin Lesion AI API")


@app.get("/")
def root():
    return {"message": "AI service is running"}


@app.post("/predict")
async def predict_endpoint(
    clinical_image: UploadFile = File(...),
    dermoscopic_image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    skin_tone: int = Form(...),
    site: str = Form(...)
):
    # Save clinical image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as c_file:
        c_file.write(await clinical_image.read())
        clinical_path = Path(c_file.name)

    # Save dermoscopic image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as d_file:
        d_file.write(await dermoscopic_image.read())
        dermoscopic_path = Path(d_file.name)

    try:
        # Prepare raw metadata
        df = pd.DataFrame([{
            "age_approx": age,
            "sex": sex,
            "skin_tone_class": skin_tone,
            "site": site
        }])

        # Convert metadata to the exact 11 columns used during training
        metadata = encode_metadata(
            df,
            columns=TRAINING_COLUMNS
        ).iloc[0].to_numpy(dtype=np.float32)

        # Run AI model
        result = predict(
            clinical_path,
            dermoscopic_path,
            metadata
        )

        return result

    finally:
        # Delete temporary images
        clinical_path.unlink(missing_ok=True)
        dermoscopic_path.unlink(missing_ok=True)