import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Concatenate,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from keras.models import Model
from keras.applications import EfficientNetB3


def build_model(metadata_size):
    clinical_input = Input(shape=(300, 300, 3), name="clinical")
    dermoscopic_input = Input(shape=(300, 300, 3), name="dermoscopic")
    metadata_input = Input(shape=(metadata_size,), name="metadata")

    backbone = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(300, 300, 3),
    )

    # İlk aşamada backbone'u dondur
    backbone.trainable = False

    clinical_features = GlobalAveragePooling2D()(backbone(clinical_input))
    derm_features = GlobalAveragePooling2D()(backbone(dermoscopic_input))

    fusion = Concatenate()([
        clinical_features,
        derm_features,
        metadata_input,
    ])

    x = BatchNormalization()(fusion)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    output = Dense(11, activation="softmax")(x)

    model = Model(
        inputs=[clinical_input, dermoscopic_input, metadata_input],
        outputs=output,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model