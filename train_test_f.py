import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

import constants
import data_import
from losses.MSE import MSE


logging.basicConfig(
    level=logging.INFO,
    filename='results/train_test_f.log',
    filemode='a',
    format='%(message)s'
)

METRICS = {
    'MSE': MSE,
    'SSIM': structural_similarity_index_measure,
    'FID': FrechetInceptionDistance}
loss = 'SSIM'


def train(model: nn.Sequential,
          train_loader: DataLoader,
          optimizer: torch.optim,) -> None:
    model.train()
    for images, _ in train_loader:
        images = images.to(constants.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        try:
            criterion = METRICS.get(loss)
        except KeyError:
            logging.error('There is no such metric {}'.format(loss))
        if loss == 'FID':
            fid = criterion(feature=2048)
            fid.update(images, real=True)
            fid.update(outputs, real=False)
            batch_loss = fid.compute()
        else:
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
            try:
                criterion = METRICS.get(loss)
            except KeyError:
                logging.error('There is no such metric {}'.format(loss))
            if loss == 'FID':
                fid = criterion(feature=2048)
                fid.update(images, real=True)
                fid.update(outputs, real=False)
                batch_test_loss = fid.compute()
            else:
                batch_test_loss = criterion(outputs, images)
            test_loss.append(batch_test_loss.item()*constants.BATCH_SIZE)
    return sum(test_loss)/len(data_import.testset)
