import os

# Render Free gibi düşük RAM'li ortamlarda TensorFlow'un
# gereksiz thread/buffer kullanımını azalt.
# Bunlar TensorFlow import edilmeden ÖNCE ayarlanmalı.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, UploadFile

from ai.src.gradcam import make_gradcam_heatmap, heatmap_to_base64


# --------------------------------------------------
# TensorFlow memory / CPU configuration
# --------------------------------------------------

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Skin Lesion Grad-CAM API",
    version="1.0.0"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

KERAS_MODEL_PATH = (
    ROOT
    / "ai"
    / "models"
    / "multimodal_model.keras"
)


# --------------------------------------------------
# Global model
# --------------------------------------------------

gradcam_model = None

IMG_SIZE = 300


# --------------------------------------------------
# Load model lazily
# --------------------------------------------------

def get_gradcam_model():

    global gradcam_model

    if gradcam_model is None:

        print(
            "DEBUG: Loading Keras model for Grad-CAM...",
            flush=True
        )

        gradcam_model = tf.keras.models.load_model(
            KERAS_MODEL_PATH,
            compile=False
        )

        gradcam_model.trainable = False

        print(
            "DEBUG: Keras Grad-CAM model loaded.",
            flush=True
        )

    return gradcam_model


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

async def preprocess_uploaded_image(
    file: UploadFile
):

    image_bytes = await file.read()

    img = tf.io.decode_jpeg(
        image_bytes,
        channels=3
    )

    img = tf.image.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        method="bilinear"
    )

    img = tf.cast(
        img,
        tf.float32
    )

    img = img / 255.0

    img = tf.expand_dims(
        img,
        axis=0
    )

    return img


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def create_metadata(
    age: float,
    sex: str,
    skin_tone: float,
    site: str
):

    metadata = np.zeros(
        (1, 11),
        dtype=np.float32
    )

    # metadata_columns.json order:
    #
    # 0  age_approx
    # 1  skin_tone_class
    # 2  sex_female
    # 3  sex_male
    # 4  site_foot
    # 5  site_genital
    # 6  site_hand
    # 7  site_head_neck_face
    # 8  site_lower_extremity
    # 9  site_trunk
    # 10 site_upper_extremity

    metadata[0, 0] = float(age)
    metadata[0, 1] = float(skin_tone)

    sex = sex.strip().lower()
    site = site.strip().lower()

    if sex == "female":
        metadata[0, 2] = 1.0

    elif sex == "male":
        metadata[0, 3] = 1.0

    site_indexes = {
        "foot": 4,
        "genital": 5,
        "hand": 6,
        "head_neck_face": 7,
        "lower_extremity": 8,
        "trunk": 9,
        "upper_extremity": 10,
    }

    site_index = site_indexes.get(site)

    if site_index is not None:
        metadata[0, site_index] = 1.0

    metadata_tensor = tf.convert_to_tensor(
        metadata,
        dtype=tf.float32
    )

    del metadata

    return metadata_tensor


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "Grad-CAM service is running"
    }


# --------------------------------------------------
# Grad-CAM endpoint
# --------------------------------------------------

@app.post("/gradcam")
async def gradcam(
    clinical_image: UploadFile = File(...),
    dermoscopic_image: UploadFile = File(...),
    age: float = Form(...),
    sex: str = Form(...),
    skin_tone: float = Form(...),
    site: str = Form(...)
):

    print(
        "DEBUG: Grad-CAM request started",
        flush=True
    )

    # -------------------------
    # Prepare images
    # -------------------------

    clinical = await preprocess_uploaded_image(
        clinical_image
    )

    dermoscopic = await preprocess_uploaded_image(
        dermoscopic_image
    )

    metadata = create_metadata(
        age=age,
        sex=sex,
        skin_tone=skin_tone,
        site=site
    )

    model = get_gradcam_model()

    # ==================================================
    # CLINICAL
    # ==================================================

    print(
        "DEBUG: Creating clinical Grad-CAM...",
        flush=True
    )

    clinical_heatmap = make_gradcam_heatmap(
        model=model,
        clinical_image=clinical,
        dermoscopic_image=dermoscopic,
        metadata=metadata,
        branch="clinical"
    )

    clinical_base64 = heatmap_to_base64(
        clinical,
        clinical_heatmap
    )

    del clinical_heatmap

    gc.collect()

    print(
        "DEBUG: Clinical Grad-CAM completed.",
        flush=True
    )

    # ==================================================
    # DERMOSCOPIC
    # ==================================================

    print(
        "DEBUG: Creating dermoscopic Grad-CAM...",
        flush=True
    )

    dermoscopic_heatmap = make_gradcam_heatmap(
        model=model,
        clinical_image=clinical,
        dermoscopic_image=dermoscopic,
        metadata=metadata,
        branch="dermoscopic"
    )

    dermoscopic_base64 = heatmap_to_base64(
        dermoscopic,
        dermoscopic_heatmap
    )

    del dermoscopic_heatmap

    # Images / metadata are no longer required
    del clinical
    del dermoscopic
    del metadata

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
        "clinical_gradcam": clinical_base64,
        "dermoscopic_gradcam": dermoscopic_base64
    }