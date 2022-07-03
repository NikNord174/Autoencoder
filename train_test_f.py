import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import constants
# from losses import SSIM
from losses import MSE


criterion = MSE


def train(model: nn.Sequential,
          train_loader: DataLoader,
          optimizer: torch.optim,
          device: torch.device,
          loss=criterion) -> None:
    model.train()
    for images, _ in train_loader:
        images = images.to(constants.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        batch_loss = criterion(outputs, images)
        batch_loss.backward()
        optimizer.step()


def test(model: nn.Sequential,
         test_loader: DataLoader,
         device: torch.device,
         loss=criterion) -> float:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(constants.DEVICE)
            output = model(images)
            loss = criterion(output, images)
            test_loss += loss.item()
    return test_loss / len(test_loader)
