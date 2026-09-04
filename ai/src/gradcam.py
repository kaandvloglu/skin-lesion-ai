
import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model=tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_output,preds=grad_model(img_array)

        class_idx=tf.argmax(preds[0])

        loss=preds[:,class_idx]

    grads=tape.gradient(loss,conv_output)

    pooled=tf.reduce_mean(grads,axis=(0,1,2))

    heatmap=conv_output[0]@pooled[...,tf.newaxis]

    heatmap=tf.squeeze(heatmap)

    heatmap=np.maximum(heatmap,0)

    heatmap/=np.max(heatmap)

    return heatmap