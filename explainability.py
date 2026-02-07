import tensorflow as tf
import numpy as np
import pandas as pd

def integrated_gradients(model, X_sample, seq_len):

    baseline = tf.zeros(shape=(1, seq_len, X_sample.shape[2]))
    input_tensor = tf.convert_to_tensor(X_sample[:1], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)

    grads = tape.gradient(preds, input_tensor)
    attributions = (input_tensor - baseline) * grads

    importance = tf.reduce_mean(tf.abs(attributions), axis=[0,1]).numpy()

    return pd.DataFrame({
        "Feature": ['Open','High','Low','Close','Volume'],
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)
