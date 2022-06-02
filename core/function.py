import os
import jax
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from functools import partial
from matplotlib import pyplot as plt

from utils.utils import AverageMeter
from core.criterion import compute_metrics


def train_torch(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, criterion, optimizer, model, lr_scheduler, 
          iter_times, writer_dict):
    model.train()

    num_steps = len(trainloader)
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels  = batch
        images = images.cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses = torch.unsqueeze(loss,0)
        loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update((epoch * num_steps + i_iter) // config.TRAIN.ACCUMULATION_STEPS)

        reduced_loss = loss

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        avg_iter_time = batch_time.average()
        iter_times['train'].append(avg_iter_time)
        avg_loss = ave_loss.average()

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.4f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      avg_iter_time, [x['lr'] for x in optimizer.param_groups], avg_loss)
            logging.info(msg)

    writer.add_scalar('train_loss', avg_loss, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    return ave_loss.average()
    
def validate_torch(config, validloader, criterion, model, iter_times, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    batch_time = AverageMeter()
    tic = time.time()

    with torch.no_grad():
        for batch in validloader:
            image, label = batch
            image = image.cuda(non_blocking=True)
            label = label.long().cuda(non_blocking=True)

            pred = model(image)

            loss = criterion(pred, label)
            losses = torch.unsqueeze(loss,0)
            loss = losses.mean()

            accuracy = torch.mean((torch.argmax(pred, -1) == label)*1.0)
            ave_acc.update(accuracy.item())

            reduced_loss = loss
            ave_loss.update(reduced_loss.item())

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            avg_iter_time = batch_time.average()
            iter_times['valid'].append(avg_iter_time)
            avg_loss = ave_loss.average()
            avg_acc = ave_acc.average()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', avg_loss, global_steps)
    writer.add_scalar('valid_acc', avg_acc, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return avg_loss, avg_acc


@partial(jax.jit, static_argnums=(3,4))
def train_step(state, batch, key, criterion, lr_scheduler):
    def loss_fn(params):
        outputs, _ = state.apply_fn(
            {'params': params},
            batch[0],
            # batch['image'],
            mutable=[],
            train=True,
            rngs={'dropout': key})
        loss = criterion(outputs, batch[1])
        # loss = criterion(outputs, batch['label'])
        return loss, outputs
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    # metrics = compute_metrics(logits=outputs, labels=batch['label'])
    lr = lr_scheduler(state.step)
    return new_state, lr, loss

@partial(jax.jit, static_argnums=(3,))
def valid_step(state, batch, key, criterion):
    def loss_fn(params):
        outputs, _ = state.apply_fn(
            {'params': params},
            batch[0],
            # batch['image'],
            mutable=[],
            train=False,
            rngs={'dropout': key})
        # loss = criterion(outputs, batch['label'])
        return outputs
    outputs = loss_fn(state.params)
    metrics = compute_metrics(logits=outputs, labels=batch[1])
    # metrics = compute_metrics(logits=outputs, labels=batch['label'])
    return metrics


def train_jax(config, state, train_ds, epoch, num_epoch, epoch_iters, criterion, 
                lr_scheduler, rng, iter_times, writer_dict):
    """Trains for a single epoch."""
    batch_metrics = []
    ave_loss = AverageMeter()
    batch_time = AverageMeter()
    tic = time.time()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(train_ds.as_numpy_iterator()):
        state, lr, loss = train_step(state, batch, rng, criterion, lr_scheduler)
        
        batch_time.update(time.time() - tic)
        tic = time.time()
        avg_iter_time = batch_time.average()
        iter_times['train'].append(avg_iter_time)

        ave_loss.update(loss.item())

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.4f}, ' \
                  'lr: {}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), lr, loss)
            logging.info(msg)

    avg_loss = ave_loss.average()
    writer.add_scalar('train_loss', avg_loss, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    return state, avg_loss, lr

def validate_jax(state, valid_ds, rng, criterion, iter_times, writer_dict):
    """Trains for a single epoch."""
    batch_metrics = []
    batch_time = AverageMeter()
    tic = time.time()
    
    for batch in valid_ds.as_numpy_iterator():
        metrics = valid_step(state, batch, rng, criterion)
        batch_metrics.append(metrics)
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        avg_iter_time = batch_time.average()
        iter_times['valid'].append(avg_iter_time)

    # compute mean of metrics across each batch in epoch.
    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics[0]}

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', epoch_metrics['loss'], global_steps)
    writer.add_scalar('valid_acc', epoch_metrics['accuracy'], global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    return epoch_metrics