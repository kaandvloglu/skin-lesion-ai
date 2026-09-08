import pandas as pd
import tensorflow as tf
import keras
from keras import layers
IMG_SIZE = 300

def preprocess_image(path):
    img = tf.io.read_file(path)
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


augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.15),
])


def augment_image(image):
    image = augmenter(image, training=True)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image