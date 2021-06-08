
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing
import cv2

from timm.utils import *

class SkinDataset(Dataset):
  def __init__(self, data_dir, df, transform=None):
    self.data_dir = data_dir
    self.df = df
    self.img_names = df['image_name'].values
    self.labels = df['benign_malignant'].values
    self.targets = preprocessing.LabelEncoder().fit_transform(self.labels)
    self.transform = transform
      
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    file_name = self.img_names[idx]
    file_path = os.path.join(self.data_dir, f'{file_name}.jpg')
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if self.transform:
      augmented = self.transform(image=image)
      image = augmented['image']
    target = torch.tensor(self.targets[idx], dtype=torch.long)
    return image, target

class LungDataset(Dataset):
  def __init__(self, df, transform=None):
    self.df = df
    self.img_names = df['image_link'].values
    self.labels = df['label'].values
    self.targets = preprocessing.LabelEncoder().fit_transform(self.labels)
    self.transform = transform
      
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    image = cv2.imread(self.img_names[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if self.transform:
      augmented = self.transform(image=image)
      image = augmented['image']
    target = torch.tensor(self.targets[idx], dtype=torch.long)
    return image, target
