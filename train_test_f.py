import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import (LR, DEVICE, BATCH_SIZE, SSIM_COEF, MSE_COEF)
import data_import

from losses.MSE import MSE
from losses.SSIM import SSIM

from models.Autoencoder_Initial import Autoencoder_Initial
from models.Autoencoder_ConvTranspose import Autoencoder_ConvTranspose
from models.Autoencoder_Upsampling import Autoencoder_Upsampling
from models.subpixel_conv import Subpixel_Conv


logging.basicConfig(
    level=logging.INFO,
    filename='results/train_test_f.log',
    filemode='a',
    format='%(message)s'
)


MODELS = {
    'Initial': Autoencoder_Initial(),
    'ConvTranspose': Autoencoder_ConvTranspose(),
    'Upsampling': Autoencoder_Upsampling(),
    'Subpixel': Subpixel_Conv(),
}

model_name = 'Subpixel'
model = MODELS.get(model_name)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train(model: nn.Sequential,
          train_loader: DataLoader,
          optimizer: torch.optim) -> None:
    model.train()
    for images, _ in train_loader:
        images = images.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        ssim = SSIM(outputs, images)
        mse = MSE(outputs, images)
        batch_loss = SSIM_COEF * ssim + MSE_COEF * mse
        batch_loss.backward()
        optimizer.step()


def test(model: nn.Sequential,
         test_loader: DataLoader,) -> float:
    model.eval()
    test_loss = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            ssim = SSIM(outputs, images)
            mse = MSE(outputs, images)
            batch_test_loss = SSIM_COEF * ssim + MSE_COEF * mse
            test_loss.append(batch_test_loss.item() * BATCH_SIZE)
            return sum(test_loss) / len(data_import.testset)
