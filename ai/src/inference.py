import json
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from .preprocessing import preprocess_image, encode_metadata
from .gradcam import make_gradcam_heatmap, heatmap_to_base64

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

    inputs = {
        "clinical": clinical,
        "dermoscopic": derm,
        "metadata": tf.convert_to_tensor(metadata),
    }

    probs = model.predict(inputs, verbose=0)[0]

    clinical_heatmap = make_gradcam_heatmap(
        model,
        inputs,
        branch="clinical",
    )

    dermoscopic_heatmap = make_gradcam_heatmap(
        model,
        inputs,
        branch="dermoscopic",
    )

    top3 = np.argsort(probs)[::-1][:3]

    print("\nPrediction")
    print("-" * 30)

    for rank, idx in enumerate(top3, 1):
        print(f"{rank}. {CLASSES[idx]:8} {probs[idx] * 100:.2f}%")

    result = {
        "prediction": CLASSES[top3[0]],
        "confidence": float(probs[top3[0]]),
        "scores": {
            CLASSES[i]: float(probs[i])
            for i in range(len(CLASSES))
        },
        "top3": [
            {
                "label": CLASSES[idx],
                "confidence": float(probs[idx]),
            }
            for idx in top3
        ],
        "gradcam": {
            "clinical": heatmap_to_base64(
                clinical,
                clinical_heatmap,
            ),
            "dermoscopic": heatmap_to_base64(
                derm,
                dermoscopic_heatmap,
            ),
        },
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--clinical", required=True)
    parser.add_argument("--dermoscopic", required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", required=True)
    parser.add_argument("--skin-tone", type=int, required=True)
    parser.add_argument("--site", required=True)

    args = parser.parse_args()

    result = predict(
        args.clinical,
        args.dermoscopic,
        args.age,
        args.sex,
        args.skin_tone,
        args.site,
    )

    print("\nReturned Result:")
    print(json.dumps(result, indent=2))