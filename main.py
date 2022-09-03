import logging
import torch
from time import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from torchsummary import summary

from utils import imshow
from data_import import train_loader, test_loader
from constants import (
    LR, DEVICE, BATCH_SIZE, EPOCHS, NO_PROGRESS_EPOCHS, MSE_COEF, SSIM_COEF)
from train_test_f import train, test, optimizer, model, model_name


logging.basicConfig(
    level=logging.INFO,
    filename='results/Experiments.log',
    filemode='a',
    format='%(message)s'
)

file = logging.getLogger(__name__)
file.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(message)s')
file_handler = RotatingFileHandler(
    'results/Experiments_main.log',
    mode='a', maxBytes=5*1024*1024,
    backupCount=2)
file_handler.setFormatter(file_formatter)
file.addHandler(file_handler)

console = logging.getLogger('console')
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console.addHandler(console_handler)


if __name__ == '__main__':
    try:
        console.info(DEVICE)
        date = datetime.now()
        file.info('Experiment: {}'.format(
            date.strftime('%m/%d/%Y, %H:%M:%S')))
        file.info('Device: {}'.format(DEVICE))
        file.info('Model: {}'.format(model_name))
        file.info('Model summary: {}'.format(
            summary(model, (3, 32, 32))))
        file.info('Model detail: {}'.format(model.__repr__()))
        file.info(f'Loss: {MSE_COEF}*MSE+{SSIM_COEF}*SSIM')
        file.info('Batch size: {}'.format(BATCH_SIZE))
        file.info('Learning rate: {}'.format(LR))
        comment = input('Comment: ')
        file.info('Comment: {}'.format(comment))
        t0 = time()
        test_loss_list = []
        n = 0
        model = model.to(DEVICE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            verbose=True
        )
        for epoch in range(EPOCHS):
            train(model, train_loader, optimizer)
            test_loss = test(model, test_loader)
            scheduler.step(test_loss)
            test_loss_list.append(round(test_loss, 5))
            t1 = (time() - t0) / 60
            msg = 'Epoch: {}, test loss: {:.5f}, time: {:.2f} min'.format(
                    epoch+1, test_loss, t1)
            file.info(msg)
            console.info(msg)
            if test_loss <= min(test_loss_list):
                n = 0
                continue
            else:
                n += 1
                if n > NO_PROGRESS_EPOCHS:
                    progress_msg = 'No progress for more than 5 epochs'
                    file.info(progress_msg)
                    console.info(progress_msg)
                    break
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Learning has been stopped manually')
    finally:
        for image, _ in test_loader:
            fake = model(image.to(DEVICE))
            imshow(image, fake, 'image', 'fake')
            break
        file.info('----------------')
