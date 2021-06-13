#!/usr/bin/env python3
""" ImageNet Training Script
This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.
This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)
NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from ImageDataset import SkinDataset, LungDataset
from util.earlystop import EarlyStopping
from loss import FocalLoss

has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Custom Arguments
parser.add_argument('--csv-path', default=None, type=str, help='Path of csv file')
parser.add_argument('--data-dir', default=None, type=str, help='Path of image folder')
parser.add_argument("--finetune", type=str, default='fc')
parser.add_argument('--augment', type=str, default='augment', help='Augment images')
parser.add_argument("--valid-every-n-batch", type=int, default=2)
parser.add_argument('--early-patience', type=str, default=5, metavar='N', help='early stop patience')
parser.add_argument('--early-value', type=float, default=0.01, metavar='N', help='early stop value')
parser.add_argument('--loss', default='CE', type=str, metavar='Loss Function', help='Loss Function')

# Dataset / Model parameters
# parser.add_argument('data_dir', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('--dataset', '-d', metavar='NAME', default='',
#                     help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='accuracy', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    #wandb.init(project=args.experiment, config=args)
    setup_default_logging()
    args, args_text = _parse_args()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
             
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
    args.world_size = 1
    args.rank = 0  # global rank
    
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    
    random_seed(args.seed, args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    model = determine_layer(model, args.finetune)
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    model.to(device)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    # amp_autocast = suppress  # do nothing
    loss_scaler = None
    
    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    
    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    # dataset_train = create_dataset(
    #     args.dataset,
    #     root=args.data_dir, split=args.train_split, is_training=True,
    #     batch_size=args.batch_size, repeats=args.epoch_repeats)
    # dataset_eval = create_dataset(
    #     args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

    # # setup mixup / cutmix
    # collate_fn = None
    # mixup_fn = None
    
    # create data loaders w/ augmentation pipeiine
    if 'skin' in args.experiment:
        _logger.info('Loading Dataset')
        img_df = pd.read_csv(args.csv_path)  # csv directory
        img_names, labels = list(img_df['image_name']), list(img_df['diagnosis'])
        img_index = list(range(len(img_names)))
        train_valid_index, _, train_valid_labels, _ = train_test_split(
            img_index, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=args.seed
        )
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for fold_index, (train_index, valid_index) in enumerate(kf.split(train_valid_index, train_valid_labels)):
            train_index = train_index
            valid_index = valid_index
            break
        
        _logger.info(f'augmentation : {args.augment}')
        train_df = img_df[img_df.index.isin(train_index)].reset_index(drop=True)
        train_dataset = SkinDataset(data_dir=args.data_dir, df=train_df, transform=get_skin_transforms(augment=args.augment, args=args))  # file directory
        valid_df = img_df[img_df.index.isin(valid_index)].reset_index(drop=True)
        valid_dataset = SkinDataset(data_dir=args.data_dir, df=valid_df, transform=get_skin_transforms(augment='none', args=args))  # file directory

        _logger.info('Load Sampler & Loader')
        print(len(train_index), len(valid_index))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
    elif 'lung' in args.experiment:
        _logger.info('Loading Dataset')
        img_df = pd.read_csv(args.csv_path) # csv directory
        img_names, labels = list(img_df['image_link']), list(img_df['label'])
        img_index = list(range(len(img_names)))
        
        _logger.info(f'augmentation : {args.augment}')
        train_df = img_df[img_df['tvt'] == 'train'].reset_index(drop=True)
        train_dataset = LungDataset(df=train_df, transform=get_lung_transforms(augment=args.augment, args=args)) # file directory
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

        valid_df = img_df[img_df['tvt'] == 'valid'].reset_index(drop=True)
        valid_dataset = LungDataset(df=valid_df, transform=get_lung_transforms(augment='none', args=args)) # file directory
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

        _logger.info('Load Sampler & Loader')
        _logger.info(len(train_dataset), len(valid_dataset))
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        valid_loader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)

    # setup loss function
    if args.smoothing:
        _logger.info('default loss is LabelSmoothing')
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
    else:
        if args.loss == 'focal':
            train_loss_fn = FocalLoss().to(device)
        else:
            train_loss_fn = nn.CrossEntropyLoss().to(device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    early_stopping = EarlyStopping(patience=args.early_patience, delta=args.early_value, verbose=True)
    if args.rank == 0:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
            str(data_config['input_size'][-1])
        ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            
            train_metrics = train_one_epoch(
                epoch, model, train_loader, optimizer, train_loss_fn, device, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir)

            eval_metrics = validate(model, valid_loader, validate_loss_fn, device, args)
            
            if lr_scheduler is not None:
                # step LR for next epoch
                if args.sched == 'step':
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
                else:
                    lr_scheduler.step(epoch)
            
            early_stopping(eval_metric, eval_metrics[eval_metric], model)
            if early_stopping.early_stop:
                _logger.info('Early Stop')
                break

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def str2bool(string):
    if string.lower() in ('true', 't'):
        return True
    elif string.lower() in ('false', 'f'):
        return False
    else:
        return string

def determine_layer(model, finetune):
    if isinstance(finetune, bool):
        if finetune:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False
    else:
        if 'last' in finetune:
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif 'head' in finetune:
            for name, param in model.named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif 'fc' in finetune:
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    return model


def get_skin_transforms(*, augment, args):
    transforms_train = A.Compose([
    A.SmallestMaxSize(max_size=args.img_size*2),
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),

    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(args.img_size, args.img_size),
    A.CoarseDropout(max_height=int(args.img_size * 0.375), max_width=int(args.img_size * 0.375), max_holes=1, p=0.7),    
    A.Normalize(),
    ToTensorV2()
    ])

    transforms_val = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(),
        ToTensorV2()
    ])

    if augment == 'augment':
        return transforms_train
    else:
        return transforms_val

def get_lung_transforms(*, augment, args):
    transforms_train = A.Compose([
    A.SmallestMaxSize(max_size=args.img_size),
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    
    A.Resize(args.img_size, args.img_size),
    
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2()
    ])

    transforms_val = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])

    if augment == 'augment':
        return transforms_train
    else:
        return transforms_val

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, device, args,
        lr_scheduler=None, saver=None, output_dir=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    #batch_time_m = AverageMeter()
    #data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    #end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in tqdm(enumerate(loader), total=len(loader)):
        last_batch = batch_idx == last_idx
        #data_time_m.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        
        loss.backward(create_graph=second_order)
        if args.clip_grad is not None:
            dispatch_clip_grad(
                model_parameters(model, exclude_head='agc' in args.clip_mode),
                value=args.clip_grad, mode=args.clip_mode)
        optimizer.step()

        #torch.cuda.synchronize()
        num_updates += 1
        #batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'LR: {lr:.3e}  '.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        lr=lr))

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, device, args, log_suffix=''):
    #batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    #end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.to(device)
            target = target.to(device) 
            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc = sum([i == j for i, j in zip(torch.argmax(output, 1).tolist(), target)]) / len(target)

            reduced_loss = loss.data

            #torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc.item(), output.size(0))

            #batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
