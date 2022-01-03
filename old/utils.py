from PIL import Image
from matplotlib import pyplot as plt
import glob
from matplotlib.lines import Line2D
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, ConcatDataset, Dataset
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchmetrics import Accuracy
import numpy as np

basic_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
train_augmenter_1 = transforms.Compose(
    [
        # transforms.CenterCrop(500),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)

train_data = datasets.ImageFolder("processed_data/train", transform=basic_transform)
train_data_1 = datasets.ImageFolder("processed_data/train", transform=train_augmenter_1)
brain_train = ConcatDataset([train_data, train_data_1])
x = DataLoader(brain_train, batch_size=128, num_workers=0, shuffle=True)
y = next(iter(x))
print(y[1])
