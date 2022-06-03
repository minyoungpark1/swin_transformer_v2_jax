import os
import wandb
import pickle
import pprint
import timeit
import logging
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tensorboardX import SummaryWriter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from data.dataset import ImageNetCustomDataset
from models.swin_transformer_v2 import SwinTransformerV2
from config.config import get_config
from core.function import train_torch, validate_torch
from core.optimizer import CosineAnnealingWarmUpRestarts, build_optimizer
from utils.utils import seed_everything, create_logger
import albumentations
import albumentations.pytorch
from timm.scheduler.cosine_lr import CosineLRScheduler

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
    transform =  albumentations.Compose([albumentations.Resize(config.DATA.IMG_SIZE,
                                                                config.DATA.IMG_SIZE),
                                        albumentations.Normalize(mean=mean, std=std),
                                        albumentations.pytorch.transforms.ToTensorV2()])

    # dataset = ImageNetCustomDataset(root=config.DATA.DATA_PATH, transform=transform)
    # n = len(dataset)
    # train_dataset = torch.utils.data.Subset(dataset, range(int(0.8*n)))
    # valid_dataset = torch.utils.data.Subset(dataset, range(int(0.8*n), n))

    train_dataset = ImageNetCustomDataset(root=os.path.join(config.DATA.DATA_PATH, 'train'), transform=transform)
    valid_dataset = ImageNetCustomDataset(root=os.path.join(config.DATA.DATA_PATH, 'val'), transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE,
                             num_workers=config.DATA.NUM_WORKERS, pin_memory=True, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=config.DATA.BATCH_SIZE,
                             num_workers=config.DATA.NUM_WORKERS, pin_memory=True, shuffle=True)


    ########## Load model ##########
    print('Loading Swin Transformer v2 model...')
    model =  SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                in_chans=config.MODEL.SWINV2.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                depths=config.MODEL.SWINV2.DEPTHS,
                                num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWINV2.APE,
                                patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES).cuda()

    ########## Load criterion, scheduler, optimizer ##########
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.BASE_LR}]
    # criterion

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = build_optimizer(config, model)

    epoch_iters = int(train_dataset.__len__() / 
                        config.DATA.BATCH_SIZE / torch.cuda.device_count())

    num_steps = int(config.TRAIN.EPOCHS * epoch_iters)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * epoch_iters)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * epoch_iters)

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        # lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=80, T_mult=1,
        #                                             eta_max=0.01, T_up=8, gamma=0.5)
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif 'lambda':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.975 ** epoch)        
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        

    best_acc = 0
    last_epoch = 0
    if config.MODEL.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_acc = checkpoint['best_acc']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            model.load_state_dict(dct)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    elif config.MODEL.PRETRAINED:
        print("Loading pretrained model...")
        PATH = config.MODEL.PRETRAINED
        pretrained_dict = torch.load(PATH)
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    start = timeit.default_timer()
    end_epoch = config.TRAIN.EPOCHS
    num_iters = config.TRAIN.EPOCHS * epoch_iters
    iter_times = {'train': [], 'valid': []}

    for epoch in range(last_epoch, end_epoch):

        train_loss = train_torch(config, epoch, config.TRAIN.EPOCHS, 
            epoch_iters, config.TRAIN.BASE_LR, num_iters,
            trainloader, criterion, optimizer, model, lr_scheduler, iter_times, writer_dict)

        valid_loss, mean_acc = validate_torch(config, validloader, criterion, model, 
        iter_times, writer_dict)

        wandb.log({
            "Learning Rate": optimizer.param_groups[0]["lr"],
            "Training Loss": train_loss,
            "Validation Loss": valid_loss,
            'Validation Accuracy': mean_acc,
        })

        logger.info('=> saving checkpoint to {}'.format(os.path.join(
            final_output_dir, 'checkpoint.pth.tar')))

        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model.state_dict(),
                    os.path.join(final_output_dir, 'best.pth'))

        torch.save({
            'epoch': epoch+1,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

        msg = 'Loss: {:.3f}, Mean Accuracy: {: 4.4f}, Best Accuracy: {: 4.4f}'.format(
                    valid_loss, mean_acc, best_acc)
        logging.info(msg)

        lr_scheduler.step(epoch)

    torch.save(model.state_dict(),
            os.path.join(final_output_dir, 'final_state.pth'))

    save_path = os.path.join(final_output_dir, 'iter_times.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(iter_times, f, protocol=4)

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % np.int((end-start)/3600))
    logger.info('Done')