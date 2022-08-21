import sys
sys.path.append('../Autoencoder')

import logging
import torch
import torch.nn as nn

from data_import import train_loader
from torchmetrics import StructuralSimilarityIndexMeasure

logging.basicConfig(
    level=logging.INFO,
    filename='losses/SSIM_real.log',
    filemode='a',
    format='%(message)s'
)


def SSIM(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    k1: float = 0.01
    k2: float = 0.03
    logging.info('fake shape {}'.format(fake.shape))
    fake = fake[:, 0, :8, :8]#*4*torch.ones([64, 1, 8, 8])
    real = real[:, 0, :8, :8]#*4*torch.ones([64, 1, 8, 8])
    logging.info('fake shape {}'.format(fake.shape))
    logging.info('min: {}, max: {}'.format(torch.min(fake), torch.max(real)))
    '''tanh = nn.Tanh()
    fake = tanh(fake)
    real = tanh(real)
    logging.info('min: {}, max: {}'.format(torch.min(fake), torch.max(real)))'''
    mean_fake = torch.mean(fake, dim=[1, 2])
    mean_real = torch.mean(real, dim=[1, 2])
    var_fake = torch.var(fake, dim=[1, 2])
    var_real = torch.var(real, dim=[1, 2])
    c1 = (k1) ** 2  # L=1, bcs image is normalized
    c2 = (k2) ** 2  # L=1, bcs image is normalized
    fake_dif = torch.sub(fake, mean_fake[:, None, None])
    real_dif = torch.sub(real, mean_real[:, None, None])
    covariance = (
        torch.sum(fake_dif * real_dif)
        / real.size()[-1] / real.size()[-2]
    )
    ssim_numerator = (
        (2 * mean_fake * mean_real + c1) * (2 * covariance + c2)
    )
    ssim_denominator = (
        ((mean_fake) ** 2 + (mean_real) ** 2 + c1)
        * ((var_fake) ** 2 + (var_real) ** 2 + c2)
    )
    ssim = ssim_numerator / ssim_denominator
    logging.info('mean_fake:')
    logging.info(torch.mean(mean_fake))
    logging.info('mean_real:')
    logging.info(torch.mean(mean_real))
    logging.info('var_fake:')
    logging.info(torch.mean(var_fake))
    logging.info('var_real:')
    logging.info(torch.mean(var_real))
    logging.info('fake_dif:')
    logging.info(torch.mean(fake_dif[0]))
    logging.info('real_dif:')
    logging.info(torch.mean(real_dif[0]))
    logging.info('covarince: {}'.format(covariance))
    logging.info('ssim numerator: {}'.format(torch.mean(ssim_numerator)))
    logging.info('ssim denominator: {}'.format(torch.mean(ssim_denominator)))
    logging.info('mean ssim: {}'.format(torch.mean(ssim)))
    return torch.mean(ssim)


if __name__ == '__main__':
    logging.info('Without using Tanh to rand torch tensor')
    image = torch.randn([64, 3, 32, 32])
    #logging.info('min: {}, max: {}'.format(torch.min(image), torch.max(image)))
    fake = torch.randn([64, 3, 32, 32])
    SSIM(fake, fake)
    ssim = StructuralSimilarityIndexMeasure()
    metric_torch = ssim(fake, fake)
    logging.info('Pytorch metrics: {}'.format(metric_torch))
    '''for images, _ in train_loader:
        logging.info('min: {}, max: {}'.format(torch.min(images), torch.max(images)))
        SSIM(images, images)
        ssim = StructuralSimilarityIndexMeasure()
        metric_torch = ssim(images, images)
        logging.info('Pytorch metrics: {}'.format(metric_torch))
        break'''
