from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .dataset import load_dataset, create_pairs
from .preprocessing import preprocess_image, encode_metadata
from .model import build_model


# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "MILK10K"
MODEL_DIR = BASE_DIR / "models"


# Load and prepare dataset
paired = create_pairs(load_dataset(DATA_PATH), DATA_PATH)
metadata = encode_metadata(paired)

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


def build_dataset(df, metadata_df):
    clinical = df["clinical_path"].values
    dermoscopic = df["dermoscopic_path"].values
    labels = df["label"].map(label_to_index).values
    metadata_values = metadata_df.loc[df.index].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices(
        (clinical, dermoscopic, metadata_values, labels)
    )

    def process(c, d, m, l):
        return (
            {
                "clinical": preprocess_image(c),
                "dermoscopic": preprocess_image(d),
                "metadata": m,
            },
            l,
        )

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(16)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# Create TensorFlow datasets
train_ds = build_dataset(train_df, metadata)
val_ds = build_dataset(val_df, metadata)

# Build model
model = build_model(metadata.shape[1])

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3  # Daha sonra artırılacak
)

# Save model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODEL_DIR / "multimodal_model.keras")

print(f"Model saved to: {MODEL_DIR / 'multimodal_model.keras'}")