import argparse
import yaml
import os
import logging
from collections import OrderedDict, defaultdict
import wandb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from ImageDataset import LungDataset
from util.metrics import AverageMeter
from util.earlystop import EarlyStopping
from loss import FocalLoss

_logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Training Config', add_help=False)
# Dataset parameters
parser.add_argument('--config-path', default=None, type=str, help='Path of config file')
parser.add_argument('--csv-path', default=None, type=str, help='Path of csv file')
parser.add_argument('--data-dir', default=None, type=str, help='Path of image folder')

# Model parameters
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N', help='number of checkpoints to keep (default: 10)')
parser.add_argument("--finetune", type=str, default='fc')
parser.add_argument('--num-classes', type=int, default=None, metavar='N', help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=299, metavar='N', help='Image patch size (default: None => model default)')
parser.add_argument('--augment', action='store_true', default=False, help='Augment images')

parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument('-j', '--workers', type=int, default=1, metavar='N', help='how many training processes to use (default: 1)')
parser.add_argument("--log-interval", type=int, default=1)
parser.add_argument("--valid-every-n-batch", type=int, default=2)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--save-best", type=str, default='loss')
parser.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC', help='Best metric (default: "top1"')
parser.add_argument("--nprocs", type=int, default=1)
parser.add_argument("--random-seed", type=int, default=1234)
parser.add_argument('--log-wandb', action='store_true', default=False, help='log training and validation metrics to wandb')
parser.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment, name of sub-folder for output')
parser.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder (default: none, current dir)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode. One of ("norm", "value", "agc")')
# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
parser.add_argument('--patience-epochs', type=int, default=0, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR', help='warmup learning rate (default: 0.0001)')
parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N', help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--decay-epochs', type=float, default=0, metavar='N', help='epoch interval to decay LR')

# Augmentation & regularization parameters
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT', help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate (default: None)')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
    cfg = yaml.safe_load(f)
    parser.set_defaults(**cfg)
args = parser.parse_args()
args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

wandb.init(project=args.experiment, config=args)

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


def save(args, epoch, train_metrics, eval_metrics, saver):
    _logger.info('saving model...')
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    if args.output is not None:
        output_dir = get_outdir(os.path.join(args.output, args.model) if args.output else './output/train', args.experiment)
        update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                       write_header=best_metric is None, log_wandb=args.log_wandb)

    if saver is not None:
        # save proper checkpoint with eval metric
        save_metric = eval_metrics[eval_metric]
        best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
    _logger.info(f'saved! {output_dir}')

def get_transforms(*, augment=args.augment, args):
  if augment:
    return A.Compose([
      A.SmallestMaxSize(max_size=args.img_size),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
      A.CenterCrop(height=args.img_size, width=args.img_size),
      A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
      A.Transpose(p=0.5),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomBrightnessContrast(p=0.5),
      A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
      A.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
      ToTensorV2(),
    ])

  else:
    return A.Compose([
      A.SmallestMaxSize(max_size=args.img_size),
      A.CenterCrop(height=args.img_size, width=args.img_size),
      A.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      ),
      ToTensorV2(),
    ])


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, device, args):
    model.train()
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    losses_m = AverageMeter()
    last_idx = len(loader) - 1
    for idx, (image, target) in tqdm(enumerate(loader), total=len(loader)):
        last_batch = idx == last_idx
        image = image.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        y_preds = model(image)
        loss = loss_fn(y_preds, target)
        losses_m.update(loss.tolist())
        loss.backward(create_graph=second_order)
        if args.clip_grad is not None:
            dispatch_clip_grad(
                model_parameters(model, exclude_head='agc' in args.clip_mode),
                value=args.clip_grad, mode=args.clip_mode)
        optimizer.step()

        if last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            avg_lr = sum(lrl)/len(lrl)
            _logger.info(
                f'epoch: {epoch}, total_loss: {losses_m.avg}, LR: {avg_lr}')
            wandb.log({"epoch": epoch, "total_loss": losses_m.avg, "LR": avg_lr})

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, device, args):
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.eval()

    last_idx = len(loader) - 1
    with torch.no_grad():
        for idx, (image, target) in tqdm(enumerate(loader), total=len(loader)):
            last_batch = idx == last_idx
            image = image.to(device)
            target = target.to(device)
            y_preds = model(image)
            loss = loss_fn(y_preds, target)
            acc = sum([i == j for i, j in zip(torch.argmax(
                y_preds, 1).tolist(), target)]) / len(target)
            reduced_loss = loss.data
            losses_m.update(reduced_loss.item())
            acc_m.update(acc)

            if last_batch:
                _logger.info(f'avg_val_loss : {losses_m.avg}, avg_accuracy : {acc_m.avg}')
                wandb.log({"avg_val_loss": losses_m.avg, "avg_accuracy": acc_m.avg})

        metrics = OrderedDict([('loss', losses_m.avg), ('accuracy', acc_m.avg)])
        return metrics

if __name__ == "__main__":
    setup_default_logging()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")

    # Load Model
    _logger.info('Loading Model')
    model = create_model(args.model,
                        pretrained=args.pretrained,
                        num_classes=args.num_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=args.drop_block,
                        global_pool=args.gp,
                        checkpoint_path=args.initial_checkpoint)
    model = model.to(device)

    # Freeze/Unfreeze Layer
    model = determine_layer(model, args.finetune)

    # Optimizer
    lr = args.lr
    if args.finetune:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
        #optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
        #optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    #lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.9, verbose=True)
    num_epochs = args.epochs
    if args.sched == 'focal':
        loss_fn = FocalLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    _logger.info(num_epochs)
    _logger.info(args.lr)

    _logger.info('Loading Dataset')
    img_df = pd.read_csv(args.csv_path) # csv directory
    img_names, labels = list(img_df['image_link']), list(img_df['label'])
    img_index = list(range(len(img_names)))

    train_df = img_df[img_df['tvt'] == 'train'].reset_index(drop=True)
    train_dataset = LungDataset(df=train_df, transform=get_transforms(augment=args.augment, args=args)) # file directory
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    valid_df = img_df[img_df['tvt'] == 'valid'].reset_index(drop=True)
    valid_dataset = LungDataset(df=valid_df, transform=get_transforms(augment=False, args=args)) # file directory
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    _logger.info('Load Sampler & Loader')
    _logger.info(len(train_dataset), len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    # Save
    output_dir = get_outdir(args.output if args.output else f'./output/train', args.model, args.experiment)
    decreasing = True if args.eval_metric == 'loss' else False
    saver = CheckpointSaver(model=model, optimizer=optimizer, args=args, checkpoint_dir=output_dir,
                            recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    # Start to Train
    _logger.info('Start to Train ...')
    try:
        model.train()
        min_val_loss = 10.0
        max_accuracy = 0.0
        early_stopping = EarlyStopping(patience=2, delta=0.0001, verbose=True)
        for epoch in range(args.epochs):
            val_losses_t = AverageMeter()
            # Train
            train_metrics = train_one_epoch(epoch=epoch, model=model, loader=train_loader,
                                            optimizer=optimizer, loss_fn=loss_fn, device=device, args=args)
            # Evaluation
            if (epoch + 1) % args.valid_every_n_batch == 0:
                eval_metrics = validate(
                    model=model, loader=valid_loader, device=device, args=args)
                avg_val_loss, avg_accuracy = eval_metrics['loss'], eval_metrics['accuracy']
                lr_scheduler.step(avg_val_loss)

                if args.save_best.lower().startswith('loss'):
                    _logger.info(f'best loss was {min_val_loss}')
                    early_stopping('loss', avg_val_loss, model)
                    if early_stopping.early_stop:
                        _logger.info('Early Stop')
                        break
                    else:
                        min_val_loss = avg_val_loss
                        _logger.info(f"* Best valid loss * {min_val_loss:4f}")
                        save(args, epoch, train_metrics, eval_metrics, saver)

                elif args.save_best.lower().startswith('acc'):
                    _logger.info(f'best accuracy was {max_accuracy}')
                    early_stopping('accuracy', max_accuracy, model)
                    if early_stopping.early_stop:
                        _logger.info('Early Stop')
                        break
                    else:
                        max_accuracy = avg_accuracy
                        _logger.info(f"* Best accuracy * {max_accuracy:4f}")
                        save(args, epoch, train_metrics, eval_metrics, saver)

    except KeyboardInterrupt:
        if args.save_best.lower().startswith('acc'):
            _logger.infot(f'Model accuracy was {max_accuracy}')
        elif args.save_best.lower().startswith('loss'):
            _logger.info(f'Model valid loss was {min_val_loss}')
        _logger.info('Bye!')
