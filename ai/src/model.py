
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Concatenate,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

def build_model(metadata_size):

    clinical_input = Input(shape=(300,300,3),name="clinical")

    dermoscopic_input = Input(shape=(300,300,3),name="dermoscopic")

    metadata_input = Input(shape=(metadata_size,),name="metadata")

    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable=False

    clinical_features = GlobalAveragePooling2D()(
        backbone(clinical_input)
    )

    derm_features = GlobalAveragePooling2D()(
        backbone(dermoscopic_input)
    )

    fusion = Concatenate()(
        [
            clinical_features,
            derm_features,
            metadata_input
        ]
    )

    x = Dense(512,activation="relu")(fusion)

    x = Dropout(0.3)(x)

    output = Dense(
        11,
        activation="softmax"
    )(x)

    model = Model(
        inputs=[
            clinical_input,
            dermoscopic_input,
            metadata_input
        ],
        outputs=output
    )

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    return model