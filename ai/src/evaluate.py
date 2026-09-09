import json
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from .preprocessing import preprocess_image, encode_metadata
from sklearn.model_selection import train_test_split
from .dataset import load_dataset, create_pairs

# -----------------------
# Dosyalar
# -----------------------
MODEL_PATH = "ai/models/multimodal_model.keras"

# Dataset yolu (train.py ile aynı)
if os.path.exists("/kaggle/input"):
    DATA_PATH = "/kaggle/input"
else:
    DATA_PATH = "ai/data/MILK10K"

model: tf.keras.Model = load_model(MODEL_PATH)  # type: ignore[assignment]

# Dataseti tekrar oluştur
dataset = load_dataset(DATA_PATH)
paired = pd.DataFrame(create_pairs(dataset, DATA_PATH))
metadata = encode_metadata(paired)

# Eğitimdeki split ile aynı
train_df, val_df = train_test_split(
    paired,
    test_size=0.2,
    random_state=42,
    stratify=paired["label"],
)

label_to_index = {
    label: i
    for i, label in enumerate(sorted(paired["label"].unique()))
}

with open("ai/models/metadata_columns.json") as f:
    cols = json.load(f)

metadata = metadata.loc[val_df.index]
metadata = metadata.reindex(columns=cols, fill_value=0)

# -----------------------
# Label mapping
# -----------------------
labels = sorted(paired["label"].unique())
label_to_idx = {
    l: i
    for i, l in enumerate(labels)
}

idx_to_label = {
    i: l
    for l, i in label_to_idx.items()
}

# -----------------------
# Dataset
# -----------------------
def generator():
    for (_, row), meta in zip(val_df.iterrows(), metadata.values):
        yield (
            {
                "clinical": preprocess_image(row["clinical_path"]),
                "dermoscopic": preprocess_image(row["dermoscopic_path"]),
                "metadata": meta.astype("float32"),
            },
            label_to_idx[row["label"]],
        )

output_signature = (
    {
        "clinical": tf.TensorSpec((300,300,3), tf.float32),
        "dermoscopic": tf.TensorSpec((300,300,3), tf.float32),
        "metadata": tf.TensorSpec((metadata.shape[1],), tf.float32),
    },
    tf.TensorSpec((), tf.int32),
)

val_ds = tf.data.Dataset.from_generator(
    generator,
    output_signature=output_signature,
).batch(16)

# -----------------------
# Prediction
# -----------------------
probs = model.predict(val_ds, verbose=1)

y_pred = np.argmax(probs, axis=1)

y_true = val_df["label"].map(label_to_idx).to_numpy(dtype=np.int32)
# -----------------------
# Metrics
# -----------------------
print("="*40)
print("FINAL EVALUATION")
print("="*40)

print(f"Macro Precision : {precision_score(y_true,y_pred,average='macro'):.4f}")
print(f"Macro Recall    : {recall_score(y_true,y_pred,average='macro'):.4f}")
print(f"Macro F1        : {f1_score(y_true,y_pred,average='macro'):.4f}")

print()

print(f"Weighted Precision : {precision_score(y_true,y_pred,average='weighted'):.4f}")
print(f"Weighted Recall    : {recall_score(y_true,y_pred,average='weighted'):.4f}")
print(f"Weighted F1        : {f1_score(y_true,y_pred,average='weighted'):.4f}")

print()

print(classification_report(
    y_true,
    y_pred,
    target_names=labels,
))

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(12,12))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False,
)

plt.tight_layout()

Path("ai/models").mkdir(parents=True, exist_ok=True)

plt.savefig(
    "ai/models/confusion_matrix.png",
    dpi=300,
)

plt.show()