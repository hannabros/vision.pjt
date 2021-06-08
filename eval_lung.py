
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

import torch
from torch.utils.data import Dataset, DataLoader

from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *

from ImageDataset import LungDataset
from util.metrics import AverageMeter

from sklearn.metrics import multilabel_confusion_matrix, classification_report

_logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Evaluation Config', add_help=False)
# Dataset parameters
parser.add_argument('--config-path', default=None, type=str, help='Path of config file')
parser.add_argument('--csv-path', default=None, type=str, help='Path of csv file')
parser.add_argument('--data-dir', default=None, type=str, help='Path of image folder')

# Model parameters
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--best-checkpoint', default='', type=str, metavar='PATH', help='Best Model Checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N', help='number of checkpoints to keep (default: 10)')
parser.add_argument("--finetune", type=str, default='fc')
parser.add_argument('--num-classes', type=int, default=None, metavar='N', help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=299, metavar='N', help='Image patch size (default: None => model default)')
parser.add_argument('--augment', type=str, default='augment', help='Augment images')

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

def get_transforms(*, augment=args.augment, args):
  if augment == 'augment':
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

if __name__ == "__main__":
    setup_default_logging()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        _logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")

    model = create_model(args.model,
                     num_classes=args.num_classes,
                     in_chans=3,
                     pretrained=args.pretrained,
                     checkpoint_path=args.checkpoint)

    model.to(device)

    img_df = pd.read_csv(args.csv_path)  # csv directory
    img_names, labels = list(img_df['image_name']), list(img_df['benign_malignant'])
    img_index = list(range(len(img_names)))
    train_valid_index, test_index, train_valid_labels, test_labels = train_test_split(
        img_index, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=args.random_seed
    )
    test_df = img_df[img_df.index.isin(test_index)].reset_index(drop=True)
    test_dataset = LungDataset(data_dir=args.data_dir, df=test_df, transform=get_transforms(augment=args.augment, args=args))  # file directory
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()

    acc_m = AverageMeter()
    pred_ids = []
    true_ids = []
    scores_ids = []
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
            scores_ids.append(y_preds)

            if last_batch:
                _logger.info(f'avg_accuracy : {acc_m.avg}')

    _logger.info(classification_report(true_ids, pred_ids))
    _logger.info(multilabel_confusion_matrix(true_ids, pred_ids))

    test_df['image_uq'] = test_df['image_name'].map(lambda x: x.split('_')[1])
    test_df['true'] = true_ids
    test_df['pred'] = pred_ids
    test_df['result'] = [[round(i, 4) for i in sc] for sc in torch.cat(scores_ids).tolist()]

    group_df = test_df.groupby(['image_uq', 'true', 'pred'])["result"].count().reset_index(name="count")
    groups = []
    for idx, row in group_df.iterrows():
        if row['true'] == row['pred']:
            same_cnt = row['count']
            total_cnt = sum(group_df[group_df['image_uq'] == row['image_uq']]['count'])
            groups.append([row['image_uq'], same_cnt, total_cnt, round(same_cnt/total_cnt, 4)])
    tile_df = pd.DataFrame(groups, columns=['image_name', 'correct', 'total', 'percent'])

    save_path = os.path.join(args.output, args.experiment)
    test_df.to_csv(os.path.join(save_path, f'result_{args.model}.csv'), index=False)
    tile_df.to_csv(os.path.join(save_path, f'tile_{args.model}.csv'), index=False)
    _logger.info(f'result saved to {save_path}')