import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_last_conv_layer(model, branch="clinical"):
    backbone = model.get_layer("efficientnetb3")

    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return backbone, layer.name

    raise ValueError("Conv2D layer bulunamadı.")


def make_gradcam_heatmap(model, inputs, branch="clinical"):
    backbone = model.get_layer("efficientnetb3")

    last_conv = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    if last_conv is None:
        raise ValueError("Conv2D layer bulunamadı.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            backbone.get_layer(last_conv).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(inputs)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(conv_output * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def heatmap_to_base64(img_tensor, heatmap):
    """
    Heatmap'i orijinal görüntünün üzerine bindirip
    Base64 PNG olarak döndürür.
    """

    image = img_tensor[0].numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.axis("off")

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close()

    return base64.b64encode(buffer.getvalue()).decode("utf-8")