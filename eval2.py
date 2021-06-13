
import argparse
import yaml
import os
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datatime import datetime

import torch
from torch.utils.data import Dataset, DataLoader

from timm.models import create_model, safe_model_name
from timm.utils import *

from ImageDataset import SkinDataset, LungDataset
from util.metrics import AverageMeter

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

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
parser.add_argument('--early-patience', type=float, default=0.01, metavar='N', help='early stop patience')
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
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
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

def main():
  setup_default_logging()
  args, args_text = _parse_args()

  if torch.cuda.is_available():
      device = torch.device("cuda")
      _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
  else:
      device = torch.device("cpu")

  model = create_model(args.model,
                    num_classes=args.num_classes,
                    in_chans=3,
                    pretrained=args.pretrained,
                    checkpoint_path=args.initial_checkpoint)

  model.to(device)

  if 'skin' in args.experiment:
    img_df = pd.read_csv(args.csv_path)  # csv directory
    img_names, labels = list(img_df['image_name']), list(img_df['diagnosis'])
    img_index = list(range(len(img_names)))
    train_valid_index, test_index, train_valid_labels, test_labels = train_test_split(
        img_index, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=args.random_seed
    )
    test_df = img_df[img_df.index.isin(test_index)].reset_index(drop=True)
    test_dataset = SkinDataset(data_dir=args.data_dir, df=test_df, transform=get_skin_transforms(augment=args.augment, args=args))  # file directory
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
  elif 'lung' in args.experiment:
    img_df = pd.read_csv(args.csv_path)  # csv directory
    img_names, labels = list(img_df['image_link']), list(img_df['label'])
    img_index = list(range(len(img_names)))
    test_df = img_df[img_df['tvt'] == 'test'].reset_index(drop=True)
    test_dataset = LungDataset(df=test_df, transform=get_lung_transforms(augment=args.augment, args=args)) # file directory
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

  model.eval()

  acc_m = AverageMeter()
  pred_ids = []
  true_ids = []
  prob_ids = []
  last_idx = len(test_loader) - 1
  with torch.no_grad():
      for batch_idx, (image, true) in tqdm(enumerate(test_loader), total=len(test_loader)):
          last_batch = batch_idx == last_idx
          image = image.to(device)
          true = true.to(device)
          y_preds = model(image)
          acc = sum([i == j for i, j in zip(torch.argmax(y_preds, 1).tolist(), true)]) / len(true)
          acc_m.update(acc)
          pred_ids.extend(torch.argmax(y_preds, 1).tolist())
          true_ids.extend(true.tolist())
          prob = torch.nn.functional.softmax(y_preds, dim=1).tolist()
          prob_ids.extend(prob)

          if last_batch:
              _logger.info(f'avg_accuracy : {acc_m.avg}')

  _logger.info(classification_report(true_ids, pred_ids))
  _logger.info(multilabel_confusion_matrix(true_ids, pred_ids))

  exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model)
        ])
  output_dir = get_outdir(args.output if args.output else './output/train', exp_name)

  if 'skin' in args.experiment:
    result_df = pd.DataFrame({'image_name': test_df['image_name'], 'true':true_ids, 'pred':pred_ids, 'probability':prob_ids})
    result_df['prob_0'] = [round(pb[0], 4) for pb in prob_ids]
    result_df['prob_1'] = [round(pb[1], 4) for pb in prob_ids]
    result_df['prob_2'] = [round(pb[2], 4) for pb in prob_ids]
    result_df['prob_3'] = [round(pb[3], 4) for pb in prob_ids]

    result_df.to_csv(os.path.join(output_dir, f'./result_{safe_model_name(args.model)}_skin.csv'), index=False)
    _logger.info(f'result saved to {output_dir}')
  
  elif 'lung' in args.experiment:
    test_df['image_uq'] = test_df['image_name'].map(lambda x: x.split('_')[1])
    test_df['true'] = true_ids
    test_df['pred'] = pred_ids
    test_df['prob_0'] = [round(pb[0], 4) for pb in prob_ids]
    test_df['prob_1'] = [round(pb[1], 4) for pb in prob_ids]
    test_df['prob_2'] = [round(pb[2], 4) for pb in prob_ids]

    group_df = test_df.groupby(['image_uq', 'true', 'pred'])["prob_0"].count().reset_index(name="count")
    groups = []
    for idx, row in group_df.iterrows():
        if row['true'] == row['pred']:
            same_cnt = row['count']
            total_cnt = sum(group_df[group_df['image_uq'] == row['image_uq']]['count'])
            groups.append([row['image_uq'], same_cnt, total_cnt, round(same_cnt/total_cnt, 4)])
    tile_df = pd.DataFrame(groups, columns=['image_name', 'correct', 'total', 'percent'])
    
    test_df.to_csv(os.path.join(output_dir, f'./result_{safe_model_name(args.model)})lung.csv'), index=False)
    tile_df.to_csv(os.path.join(output_dir, f'./tile_{safe_model_name(args.model)}_lung.csv'), index=False)
    _logger.info(f'result saved to {output_dir}')

if __name__ == "__main__":
  main()