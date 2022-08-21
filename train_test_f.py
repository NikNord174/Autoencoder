import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

import constants
from losses.MSE import MSE


criterion = structural_similarity_index_measure


def train(model: nn.Sequential,
          train_loader: DataLoader,
          optimizer: torch.optim,) -> None:
    model.train()
    for images, _ in train_loader:
        images = images.to(constants.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        if criterion == FrechetInceptionDistance:
            fid = criterion(feature=2048)
            fid.update(images, real=True)
            fid.update(outputs, real=False)
            batch_loss = fid.compute()
        elif criterion == structural_similarity_index_measure:
            batch_loss = criterion(outputs, images, k1=0.3, k2=0.3)
        elif criterion == MSE:
            batch_loss = criterion(outputs, images)
        batch_loss.backward()
        optimizer.step()


def test(model: nn.Sequential,
         test_loader: DataLoader,) -> float:
    model.eval()
    test_loss = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(constants.DEVICE)
            outputs = model(images)
            if criterion == FrechetInceptionDistance:
                fid = criterion(feature=2048)
                fid.update(images, real=True)
                fid.update(outputs, real=False)
                loss = fid.compute()
            elif criterion == structural_similarity_index_measure:
                loss = criterion(outputs, images, k1=0.3, k2=0.3)
            elif criterion == MSE:
                loss = criterion(outputs, images)
            test_loss.append(loss.item())
    return sum(test_loss)/len(test_loss)
