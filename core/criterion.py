import torch
import jax
from jax import numpy as jnp


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    if torch.is_tensor(labels):
        x = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    else:
        x = (labels[..., None] == jnp.arange(num_classes)[None])
        x = jax.lax.select(
            x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
        x = x.astype(jnp.float32)
    return x

def cross_entropy_loss(logits, labels):
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(logp * onehot(labels, num_classes=logits.shape[-1]), axis=1))

def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics