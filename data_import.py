import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import (
    Compose, ToTensor, RandomHorizontalFlip, ColorJitter)

import constants


DOWNLOAD = False
if not os.path.exists('data/cifar-10-python.tar.gz'):
    DOWNLOAD = True

transform = Compose([
    # RandomHorizontalFlip(p=0.5),
    # ColorJitter(brightness=.5),
    ToTensor()])
trainset = datasets.CIFAR10(
    'data',
    train=True,
    download=DOWNLOAD,
    transform=transform
)
testset = datasets.CIFAR10(
    'data',
    train=False,
    download=DOWNLOAD,
    transform=transform
)
train_loader = DataLoader(
    trainset,
    batch_size=constants.BATCH_SIZE,
    num_workers=constants.NUM_WORKERS,
    shuffle=True
)
test_loader = DataLoader(
    testset,
    batch_size=constants.BATCH_SIZE,
    num_workers=constants.NUM_WORKERS,
    shuffle=False
)
