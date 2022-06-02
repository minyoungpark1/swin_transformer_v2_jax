import os
import wandb
import pickle
import pprint
import timeit
import logging
import argparse
import numpy as np

import jax
from jax import random, numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import tensorflow as tf
from tensorboardX import SummaryWriter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.swin_transformer_jax import SwinTransformerV2
from config.config import get_config
from core.criterion import cross_entropy_loss
from core.optimizer import build_optimizer_jax
from core.scheduler import build_scheduler_jax
from core.function import train_jax, validate_jax
from utils.data_utils import transform_tf_dataset
from utils.utils import seed_everything, create_logger
import albumentations


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    # parser.add_argument('--output', default='output', type=str, metavar='PATH',
    #                     help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
   
    parser.add_argument('--seed', type=int,
                        default=304, help='random seed')
    parser.add_argument('--name', type=str, required=True, help='wandb name')
                        
    args = parser.parse_args()
    config = get_config(args)
    wandb.init(project="jax_exercise", name=args.name)
    wandb.config.update(args)
    seed_everything(args.seed)

    ########## Logger generation ##########
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    ########## Dataloader generation ##########
    print("Creating dataloaders...")

    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    ## TODO ##
    transform =  albumentations.Compose([
        albumentations.Normalize(mean=mean, std=std),
                                        ])

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(config.DATA.DATA_PATH, 'train'), 
        image_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        seed=config.SEED, batch_size=config.DATA.BATCH_SIZE, shuffle=True)

    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(config.DATA.DATA_PATH, 'val'),
        image_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        seed=config.SEED, batch_size=config.DATA.BATCH_SIZE, shuffle=True)

    trainloader = transform_tf_dataset(train_ds, transform)
    validloader = transform_tf_dataset(valid_ds, transform)
    ####

    ########## Load model ##########
    print('Loading Swin Transformer v2 model...')
    model =  SwinTransformerV2(img_size=(config.DATA.IMG_SIZE,config.DATA.IMG_SIZE),
                                patch_size=(config.MODEL.SWINV2.PATCH_SIZE,config.MODEL.SWINV2.PATCH_SIZE),
                                in_chans=config.MODEL.SWINV2.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                depths=config.MODEL.SWINV2.DEPTHS,
                                num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                window_size=(config.MODEL.SWINV2.WINDOW_SIZE,config.MODEL.SWINV2.WINDOW_SIZE),
                                mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                patch_norm=config.MODEL.SWINV2.PATCH_NORM)

    ########## Load criterion, scheduler, optimizer ##########
    key1, key2 = random.split(random.PRNGKey(0), 2)
    init_rngs = {'params': key1, 'dropout': key2}
    batch = next(iter(trainloader))
    params = model.init(init_rngs, jnp.array(batch[0]), train=True)['params']
    # params = model.init(init_rngs, jnp.array(batch['image']), train=True)['params']
    
    key1, key2 = random.split(random.PRNGKey(0), 2)
    epoch_iters = len(trainloader) // jax.device_count()
    lr_scheduler = build_scheduler_jax(config, epoch_iters)
    optimizer = build_optimizer_jax(config)
    criterion = cross_entropy_loss
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        
    best_acc = 0
    last_epoch = 0
    if config.MODEL.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pkl')
        if os.path.exists(final_output_dir):
            with open(model_state_file, 'rb') as fr:
                checkpoint = pickle.load(fr)
            best_acc = checkpoint['best_acc']
            last_epoch = checkpoint['epoch']
            state = checkpoints.restore_checkpoint(ckpt_dir=model_state_file, 
                                                    target=state,
                                                    step=last_epoch,
                                                    prefix='checkpoint')

            logger.info("=> loaded checkpoint (epoch {})"
                        .format(last_epoch))

    # TODO: code is not versatile
    elif config.MODEL.PRETRAINED:
        print("Loading pretrained model...")
        PATH = config.MODEL.PRETRAINED
        filename = os.path.basename(PATH)
        state = checkpoints.restore_checkpoint(ckpt_dir=os.path.dirname(PATH), 
        target=state, prefix=filename.split('_')[0], step=filename.split('_')[1])

    start = timeit.default_timer()
    end_epoch = config.TRAIN.EPOCHS
    num_iters = config.TRAIN.EPOCHS * epoch_iters
    iter_times = {'train': [], 'valid': []}

    for epoch in range(last_epoch, end_epoch):
        state, mean_loss, lr = train_jax(config, state, trainloader, epoch, 
                                config.TRAIN.EPOCHS, epoch_iters, criterion, lr_scheduler,
                                key1, iter_times, writer_dict)
        valid_metrics = validate_jax(state, validloader, key1, criterion, iter_times, writer_dict)

        wandb.log({
            "Learning Rate": lr,
            "Training Loss": mean_loss,
            "Validation Loss": valid_metrics['loss'],
            'Validation Accuracy': valid_metrics['accuracy'],
        })

        logger.info('=> saving checkpoint to {}'.format(os.path.join(
            final_output_dir, 'checkpoint.pkl')))

        if valid_metrics['accuracy'] > best_acc:
            best_acc = valid_metrics['accuracy']
            checkpoints.save_checkpoint(ckpt_dir=final_output_dir, 
                                        target=state, 
                                        step=epoch+1,
                                        prefix='best',
                                        overwrite=True
                                        )

        checkpoints.save_checkpoint(ckpt_dir=final_output_dir, 
                                    target=state, 
                                    step=epoch+1, 
                                    prefix='checkpoint',
                                    overwrite=True
                                    )

        save_path = os.path.join(final_output_dir, 'checkpoint.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({'epoch': epoch+1, 'best_acc': best_acc}, f, protocol=4)

        msg = 'Loss: {:.3f}, Mean Accuracy: {: 4.4f}, Best Accuracy: {: 4.4f}'.format(
                    valid_metrics['loss'], valid_metrics['accuracy'], best_acc)
        logging.info(msg)

    checkpoints.save_checkpoint(ckpt_dir=final_output_dir, 
                                target=state, 
                                step=epoch+1, 
                                prefix='final_state',
                                overwrite=True
                                )

    save_path = os.path.join(final_output_dir, 'iter_times.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(iter_times, f, protocol=4)

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')