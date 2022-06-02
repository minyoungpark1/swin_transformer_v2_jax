import numpy as np
import torch
from jax import numpy as jnp
import jax
import tensorflow as tf
from functools import partial
import albumentations

AUTOTUNE = tf.data.experimental.AUTOTUNE

def transform_tf_dataset(data, transform):
    def aug_fn(image):
        data = {"image":image}
        aug_data = transform(**data)
        aug_img = aug_data["image"]
        return aug_img

    def process_data(image, label):
        aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
        return aug_img, label
    ds_alb = data.map(partial(process_data),
                    num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds_alb
    