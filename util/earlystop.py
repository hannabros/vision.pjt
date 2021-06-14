import numpy as np

import logging

_logger = logging.getLogger()

class EarlyStopping:
  """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
  def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
    """
    Args:
        patience (int): validation loss가 개선된 후 기다리는 기간
                        Default: 7
        verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                        Default: False
        delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                        Default: 0
        path (str): checkpoint저장 경로
                        Default: 'checkpoint.pt'
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.val_acc_max = np.Inf
    self.delta = delta
    self.path = path

  def __call__(self, val_criterion, value, model):
    if 'loss' == val_criterion:
      val_loss = value
      score = -val_loss

      if self.best_score is None:
        self.best_score = score
      elif score < self.best_score + self.delta:
        self.counter += 1
        _logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
          self.early_stop = True
      else:
        self.best_score = score
        self.counter = 0
    
    elif 'accuracy' == val_criterion:
      acc = value
      score = acc
      if self.best_score is None:
        self.best_score = score
      elif score < self.best_score + self.delta:
        self.counter += 1
        _logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
          self.early_stop = True
      else:
        self.best_score = score
        self.counter = 0