import base64
import io
import gc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Feature extractor only needs to be created once.
_feature_extractor = None
_feature_extractor_model_id = None


def get_feature_extractor(model):
    global _feature_extractor
    global _feature_extractor_model_id

    current_model_id = id(model)

    if (
        _feature_extractor is None
        or _feature_extractor_model_id != current_model_id
    ):
        backbone = model.get_layer("efficientnetb3")

        last_conv = None

        for layer in reversed(backbone.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        if last_conv is None:
            raise ValueError(
                "No Conv2D layer found in efficientnetb3."
            )

        _feature_extractor = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                backbone.get_layer(last_conv).output,
                backbone.output,
            ],
        )

        _feature_extractor_model_id = current_model_id

    return _feature_extractor


def heatmap_to_base64(img_tensor, heatmap):
    img = img_tensor[0].numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )
    plt.axis("off")

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    result = base64.b64encode(
        buf.getvalue()
    ).decode()

    buf.close()

    return result


def make_gradcam_heatmap(
    model,
    inputs,
    branch="clinical"
):
    feature_extractor = get_feature_extractor(model)

    if branch == "clinical":
        img = inputs["clinical"]
        other = inputs["dermoscopic"]

        gap_layer = model.get_layer(
            "global_average_pooling2d_2"
        )

        other_gap = model.get_layer(
            "global_average_pooling2d_3"
        )

    else:
        img = inputs["dermoscopic"]
        other = inputs["clinical"]

        gap_layer = model.get_layer(
            "global_average_pooling2d_3"
        )

        other_gap = model.get_layer(
            "global_average_pooling2d_2"
        )

    # ------------------------------------------------
    # IMPORTANT MEMORY OPTIMIZATION
    #
    # We do NOT need gradients for the other image.
    # Therefore calculate it completely OUTSIDE
    # GradientTape.
    # ------------------------------------------------

    _, other_feature = feature_extractor(
        other,
        training=False
    )

    other_feature = tf.stop_gradient(
        other_feature
    )

    pooled_other = other_gap(
        other_feature
    )

    del other_feature

    gc.collect()

    # ------------------------------------------------
    # GradientTape now only tracks the image for which
    # Grad-CAM is being generated.
    # ------------------------------------------------

    with tf.GradientTape() as tape:

        conv_output, feature_map = feature_extractor(
            img,
            training=False
        )

        pooled = gap_layer(
            feature_map
        )

        if branch == "clinical":
            merged = model.get_layer(
                "concatenate_1"
            )(
                [
                    pooled,
                    pooled_other,
                    inputs["metadata"],
                ]
            )

        else:
            merged = model.get_layer(
                "concatenate_1"
            )(
                [
                    pooled_other,
                    pooled,
                    inputs["metadata"],
                ]
            )

        x = model.get_layer(
            "dense_2"
        )(merged)

        x = model.get_layer(
            "dropout_1"
        )(
            x,
            training=False
        )

        preds = model.get_layer(
            "dense_3"
        )(x)

        class_idx = tf.argmax(
            preds[0]
        )

        loss = preds[
            :,
            class_idx
        ]

    grads = tape.gradient(
        loss,
        conv_output
    )

    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradient calculation failed."
        )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    result = heatmap.numpy()

    del grads
    del pooled_grads
    del conv_output
    del feature_map
    del pooled
    del pooled_other
    del preds
    del x
    del merged

    gc.collect()

    return result