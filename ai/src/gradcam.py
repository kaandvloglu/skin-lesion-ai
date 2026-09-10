import base64
import gc
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# MODEL LAYER NAMES
# ============================================================

EFFICIENTNET_LAYER = "efficientnetb3"

CLINICAL_GAP_LAYER = "global_average_pooling2d_2"
DERMOSCOPIC_GAP_LAYER = "global_average_pooling2d_3"

CONCAT_LAYER = "concatenate_1"

DENSE_LAYER = "dense_2"
DROPOUT_LAYER = "dropout_1"
OUTPUT_LAYER = "dense_3"


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    model,
    clinical_image,
    dermoscopic_image,
    metadata,
    branch="clinical",
    class_index=None
):

    """
    Memory optimized Grad-CAM.

    Critical optimization:

    EfficientNet forward passes are executed OUTSIDE
    GradientTape.

    Therefore TensorFlow does not need to keep all
    EfficientNet intermediate activations in memory
    for backpropagation.

    GradientTape only watches the final EfficientNet
    feature map and the small classification head.
    """

    if branch not in (
        "clinical",
        "dermoscopic"
    ):
        raise ValueError(
            "branch must be 'clinical' or 'dermoscopic'"
        )

    # --------------------------------------------------------
    # Get model components
    # --------------------------------------------------------

    efficientnet = model.get_layer(
        EFFICIENTNET_LAYER
    )

    clinical_gap = model.get_layer(
        CLINICAL_GAP_LAYER
    )

    dermoscopic_gap = model.get_layer(
        DERMOSCOPIC_GAP_LAYER
    )

    concatenate = model.get_layer(
        CONCAT_LAYER
    )

    dense = model.get_layer(
        DENSE_LAYER
    )

    dropout = model.get_layer(
        DROPOUT_LAYER
    )

    output_layer = model.get_layer(
        OUTPUT_LAYER
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # EfficientNet is NOT executed inside GradientTape.
    #
    # This is the major RAM optimization.
    # --------------------------------------------------------

    clinical_features = efficientnet(
        clinical_image,
        training=False
    )

    dermoscopic_features = efficientnet(
        dermoscopic_image,
        training=False
    )

    # --------------------------------------------------------
    # Select target feature map
    # --------------------------------------------------------

    if branch == "clinical":

        target_features = clinical_features

    else:

        target_features = dermoscopic_features

    # --------------------------------------------------------
    # Gradient only through classification head
    # --------------------------------------------------------

    with tf.GradientTape(
        watch_accessed_variables=False
    ) as tape:

        tape.watch(
            target_features
        )

        clinical_vector = clinical_gap(
            clinical_features
        )

        dermoscopic_vector = dermoscopic_gap(
            dermoscopic_features
        )

        combined = concatenate(
            [
                clinical_vector,
                dermoscopic_vector,
                metadata
            ]
        )

        x = dense(
            combined
        )

        x = dropout(
            x,
            training=False
        )

        predictions = output_layer(
            x
        )

        # Automatically use highest probability class
        if class_index is None:

            selected_class = tf.argmax(
                predictions[0]
            )

        else:

            selected_class = tf.cast(
                class_index,
                tf.int64
            )

        class_score = tf.gather(
            predictions[0],
            selected_class
        )

    # --------------------------------------------------------
    # Gradient of class score with respect to
    # EfficientNet output feature map
    # --------------------------------------------------------

    gradients = tape.gradient(
        class_score,
        target_features
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradient could not be calculated."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    # Remove batch dimension
    feature_map = target_features[0]

    # Weight each feature channel
    weighted_features = (
        feature_map
        * pooled_gradients
    )

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize 0 -> 1
    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    # Convert to NumPy before cleaning TensorFlow objects
    heatmap_numpy = heatmap.numpy().astype(
        np.float32
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del gradients
    del pooled_gradients
    del feature_map
    del weighted_features
    del heatmap
    del clinical_vector
    del dermoscopic_vector
    del combined
    del x
    del predictions
    del class_score
    del clinical_features
    del dermoscopic_features
    del target_features

    gc.collect()

    return heatmap_numpy


# ============================================================
# HEATMAP -> PNG -> BASE64
# ============================================================

def heatmap_to_base64(
    original_image,
    heatmap
):

    """
    Creates an overlay without Matplotlib.

    Pillow + NumPy are much lighter than importing
    matplotlib and building figures/font caches.
    """

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    image = original_image[0].numpy()

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image_uint8 = (
        image * 255.0
    ).astype(
        np.uint8
    )

    height = image_uint8.shape[0]
    width = image_uint8.shape[1]

    # --------------------------------------------------------
    # Resize heatmap to original image size
    # --------------------------------------------------------

    heatmap_uint8 = (
        np.clip(
            heatmap,
            0.0,
            1.0
        )
        * 255.0
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8,
        mode="L"
    )

    heatmap_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    resized_heatmap = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Lightweight heat colors
    #
    # Low    = transparent
    # Medium = red
    # High   = yellow
    # --------------------------------------------------------

    red = np.clip(
        resized_heatmap * 2.0,
        0.0,
        1.0
    )

    green = np.clip(
        (resized_heatmap - 0.5) * 2.0,
        0.0,
        1.0
    )

    blue = np.zeros_like(
        resized_heatmap
    )

    colored_heatmap = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    original_float = (
        image_uint8.astype(np.float32)
        / 255.0
    )

    # Stronger activation = stronger overlay
    alpha = (
        resized_heatmap[..., np.newaxis]
        * 0.55
    )

    overlay = (
        original_float * (1.0 - alpha)
        + colored_heatmap * alpha
    )

    overlay = np.clip(
        overlay * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # PNG -> Base64
    # --------------------------------------------------------

    output_image = Image.fromarray(
        overlay,
        mode="RGB"
    )

    buffer = BytesIO()

    output_image.save(
        buffer,
        format="PNG",
        optimize=False
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    buffer.close()

    del image
    del image_uint8
    del heatmap_uint8
    del heatmap_image
    del resized_heatmap
    del red
    del green
    del blue
    del colored_heatmap
    del original_float
    del alpha
    del overlay
    del output_image

    gc.collect()

    return encoded