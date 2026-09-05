import pandas as pd
import tensorflow as tf

IMG_SIZE = 300

def preprocess_image(path):

    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    return img


def encode_metadata(df, columns=None):

    metadata = df[
        [
            "age_approx",
            "sex",
            "skin_tone_class",
            "site"
        ]
    ].copy()

    metadata = pd.get_dummies(metadata)

    metadata = metadata.fillna(0)

    if columns is not None:
        metadata = metadata.reindex(columns=columns, fill_value=0)

    return metadata.astype("float32")