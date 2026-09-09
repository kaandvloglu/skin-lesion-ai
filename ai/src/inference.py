
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from .preprocessing import preprocess_image, encode_metadata

CLASSES = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]


def build_metadata(age, sex, skin_tone, site):
    row = pd.DataFrame(
        [
            {
                "age_approx": age,
                "sex": sex,
                "skin_tone_class": skin_tone,
                "site": site,
            }
        ]
    )

    metadata = encode_metadata(row)

    with open("ai/models/metadata_columns.json") as f:
        saved_columns = json.load(f)

    metadata = metadata.reindex(columns=saved_columns, fill_value=0)

    return metadata.astype("float32").values


def predict(clinical_path, dermoscopic_path, age, sex, skin_tone, site):
    model = load_model("ai/models/multimodal_model.keras")

    clinical = preprocess_image(clinical_path)
    derm = preprocess_image(dermoscopic_path)

    clinical = tf.expand_dims(clinical, 0)
    derm = tf.expand_dims(derm, 0)

    metadata = build_metadata(age, sex, skin_tone, site)

    probs = model.predict(
        {
            "clinical": clinical,
            "dermoscopic": derm,
            "metadata": metadata,
        },
        verbose=0,
    )[0]

    top3 = np.argsort(probs)[::-1][:3]

    print("\nPrediction")
    print("-" * 30)

    for rank, idx in enumerate(top3, 1):
        print(f"{rank}. {CLASSES[idx]:8} {probs[idx]*100:.2f}%")

    return CLASSES[top3[0]], probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--clinical", required=True)
    parser.add_argument("--dermoscopic", required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", required=True)
    parser.add_argument("--skin-tone", type=int, required=True)
    parser.add_argument("--site", required=True)

    args = parser.parse_args()

    predict(
        args.clinical,
        args.dermoscopic,
        args.age,
        args.sex,
        args.skin_tone,
        args.site,
    )