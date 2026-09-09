import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def heatmap_to_base64(img_tensor, heatmap):
    img = img_tensor[0].numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode()


def make_gradcam_heatmap(model, inputs, branch="clinical"):
    backbone = model.get_layer("efficientnetb3")

    last_conv = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    feature_extractor = tf.keras.Model(
        backbone.input,
        [
            backbone.get_layer(last_conv).output,
            backbone.output,
        ],
    )

    if branch == "clinical":
        img = inputs["clinical"]
        other = inputs["dermoscopic"]
        gap_layer = model.get_layer("global_average_pooling2d_2")
        other_gap = model.get_layer("global_average_pooling2d_3")
    else:
        img = inputs["dermoscopic"]
        other = inputs["clinical"]
        gap_layer = model.get_layer("global_average_pooling2d_3")
        other_gap = model.get_layer("global_average_pooling2d_2")

    with tf.GradientTape() as tape:
        conv_output, feature_map = feature_extractor(img)

        tape.watch(conv_output)

        _, other_feature = feature_extractor(other)
        other_feature = tf.stop_gradient(other_feature)

        pooled = gap_layer(feature_map)
        pooled_other = other_gap(other_feature)

        if branch == "clinical":
            merged = model.get_layer("concatenate_1")(
                [pooled, pooled_other, inputs["metadata"]]
            )
        else:
            merged = model.get_layer("concatenate_1")(
                [pooled_other, pooled, inputs["metadata"]]
            )

        x = model.get_layer("dense_2")(merged)
        x = model.get_layer("dropout_1")(x, training=False)
        preds = model.get_layer("dense_3")(x)

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()