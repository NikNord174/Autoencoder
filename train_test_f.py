import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

import constants
import data_import

from losses.MSE import MSE

from models.Autoencoder_Initial import Autoencoder_Initial
from models.Autoencoder_ConvTranspose import Autoencoder_ConvTranspose
from models.Autoencoder_Upsampling import Autoencoder_Upsampling


logging.basicConfig(
    level=logging.INFO,
    filename='results/train_test_f.log',
    filemode='a',
    format='%(message)s'
)


MODELS = {
    'Initial': Autoencoder_Initial(),
    'ConvTranspose': Autoencoder_ConvTranspose(),
    'Upsampling': Autoencoder_Upsampling()
}

model_name = 'Upsampling'
model = MODELS.get(model_name)

METRICS = {
    'MSE': MSE,
    'SSIM': structural_similarity_index_measure,
    'FID': FrechetInceptionDistance}

loss = 'MSE'
optimizer = torch.optim.Adam(model.parameters(), lr=constants.LR)


def train(model: nn.Sequential,
          train_loader: DataLoader,) -> None:
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
