from pathlib import Path
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .dataset import load_dataset, create_pairs
from .preprocessing import preprocess_image, encode_metadata
from .model import build_model

# Dataset yolu
DATA_PATH = "data/MILK10K"

# Dataseti yükle
paired = create_pairs(load_dataset(DATA_PATH), DATA_PATH)

# Metadata encode
metadata = encode_metadata(paired)

# Train / Validation split
train_df, val_df = train_test_split(
    paired,
    test_size=0.2,
    random_state=42,
    stratify=paired["label"]
)

# Label mapping
label_to_index = {
    label: i
    for i, label in enumerate(sorted(paired["label"].unique()))
}

def build_dataset(df, meta):

    clinical = df["clinical_path"].values
    derm = df["dermoscopic_path"].values
    labels = df["label"].map(label_to_index).values
    meta_values = meta.loc[df.index].values

    ds = tf.data.Dataset.from_tensor_slices(
        (
            clinical,
            derm,
            meta_values,
            labels
        )
    )

    def process(c, d, m, l):
        return (
            {
                "clinical": preprocess_image(c),
                "dermoscopic": preprocess_image(d),
                "metadata": m
            },
            l
        )

    ds = ds.map(
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(16)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

# Datasetleri oluştur
train_ds = build_dataset(train_df, metadata)
val_ds = build_dataset(val_df, metadata)

# Modeli oluştur
model = build_model(metadata.shape[1])

# Model klasörü
Path("ai/models").mkdir(parents=True, exist_ok=True)

# Callbackler
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "ai/models/multimodal_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# Eğitim
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# Son modeli kaydet
model.save("ai/models/multimodal_model.keras")

# Metadata sütunlarını kaydet
with open("ai/models/metadata_columns.json", "w") as f:
    json.dump(metadata.columns.tolist(), f)