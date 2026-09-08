import json
from pathlib import Path
import os 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .dataset import load_dataset, create_pairs
from .preprocessing import preprocess_image, augment_image, encode_metadata
from .model import build_model


# Dataset yolu (Kaggle veya lokal)
if os.path.exists("/kaggle/input"):
    metadata_file = next(Path("/kaggle/input").rglob("MILK10K_Training_Metadata.csv"))
    DATA_PATH = str(metadata_file.parent)
else:
    DATA_PATH = "data/MILK10K"

print("Using DATA_PATH:", DATA_PATH)


# Dataseti yükle
paired = create_pairs(load_dataset(DATA_PATH), DATA_PATH)

# Metadata encode
metadata = encode_metadata(paired)

# Train / Validation split
train_df, val_df = train_test_split(
    paired,
    test_size=0.2,
    random_state=42,
    stratify=paired["label"],
)

# Label mapping
label_to_index = {
    label: i
    for i, label in enumerate(sorted(paired["label"].unique()))
}

train_labels = train_df["label"].map(label_to_index).values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)

class_weights = {
    i: w
    for i, w in enumerate(weights)
}


def build_dataset(df, meta, training=False):
    clinical = df["clinical_path"].values
    derm = df["dermoscopic_path"].values
    labels = df["label"].map(label_to_index).values
    meta_values = meta.loc[df.index].values

    ds = tf.data.Dataset.from_tensor_slices(
        (
            clinical,
            derm,
            meta_values,
            labels,
        )
    )

    def process(c, d, m, l):
        clinical_img = preprocess_image(c)
        derm_img = preprocess_image(d)

        if training:
            clinical_img = augment_image(clinical_img)
            derm_img = augment_image(derm_img)

        return (
            {
                "clinical": clinical_img,
                "dermoscopic": derm_img,
                "metadata": m,
            },
            l,
        )

    ds = ds.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# Datasetleri oluştur
train_ds = build_dataset(
    train_df,
    metadata,
    training=True,
)

val_ds = build_dataset(
    val_df,
    metadata,
    training=False,
)


# Modeli oluştur
model = build_model(metadata.shape[1])


# Model klasörü
Path("ai/models").mkdir(parents=True, exist_ok=True)


# Callbackler
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    ),
    ModelCheckpoint(
        "ai/models/multimodal_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
]


# Eğitim
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights,
)


# Son modeli kaydet
model.save("ai/models/multimodal_model.keras")


# Metadata sütunlarını kaydet
with open("ai/models/metadata_columns.json", "w") as f:
    json.dump(metadata.columns.tolist(), f)