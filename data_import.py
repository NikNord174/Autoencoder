from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize

import constants


transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(
    '../data',
    train=True,
    download=False,
    transform=transform
)
testset = datasets.CIFAR10(
    '../data',
    train=False,
    download=False,
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
    shuffle=True
)
