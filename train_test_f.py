import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure
from skimage.metrics import structural_similarity as ssim
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
    'SSIM_skimage': ssim,
    'FID': FrechetInceptionDistance}

loss = 'SSIM_skimage'
optimizer = torch.optim.Adam(model.parameters(), lr=constants.LR)


def train(model: nn.Sequential,
          train_loader: DataLoader,
          optimizer: torch.optim) -> None:
    model.train()
    for images, _ in train_loader:
        images = images.to(constants.DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        try:
            criterion = METRICS.get(loss)
            if loss == 'FID':
                fid = criterion(feature=2048)
                fid.update(images, real=True)
                fid.update(outputs, real=False)
                batch_loss = fid.compute()
            else:
                batch_loss = 1 - (torch.tensor(criterion(
                    outputs.detach().numpy(), images.detach().numpy(),
                    data_range=images.max().item() - images.min().item(),
                    channel_axis=1), requires_grad=True) + 1) / 2
            batch_loss.backward()
            optimizer.step()
        except KeyError:
            logging.error('There is no such metric {}'.format(loss))


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
                if loss == 'FID':
                    fid = criterion(feature=2048)
                    fid.update(images, real=True)
                    fid.update(outputs, real=False)
                    batch_test_loss = fid.compute()
                else:
                    batch_test_loss = 1 - (criterion(
                        outputs.detach().numpy(), images.detach().numpy(),
                        data_range=images.max().item() - images.min().item(),
                        channel_axis=1) + 1) / 2
                test_loss.append(batch_test_loss.item()*constants.BATCH_SIZE)
                return sum(test_loss)/len(data_import.testset)
            except KeyError:
                logging.error('There is no such metric {}'.format(loss))
