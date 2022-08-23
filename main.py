import logging
from logging.handlers import RotatingFileHandler
from time import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
from torchsummary import summary

import constants
import train_test_f
from train_test_f import train, test
from data_import import train_loader, test_loader


logging.basicConfig(
    level=logging.INFO,
    filename='results/Experiments.log',
    filemode='a',
    format='%(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    'results/Experiments_main.log',
    mode='a', maxBytes=5*1024*1024,
    backupCount=2)
logger.addHandler(handler)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)


def np_and_save(image, image_name):
    image_cpu = image[0].detach().cpu()
    save_image(image_cpu, f'results/{image_name}.jpeg')
    image_np = image_cpu.numpy()
    return np.transpose(image_np, (1, 2, 0))


def illustration(fig, no, image, title, fontsize=28):
    ax = fig.add_subplot(1, 2, no)
    ax.set_title(title, fontsize=fontsize)
    ax.imshow(image)
    plt.axis('off')


def imshow(image, fake, image_name, fake_name):
    image_np = np_and_save(image, image_name)
    fake_np = np_and_save(fake, fake_name)
    fig = plt.figure(figsize=(15, 10))
    illustration(fig, 1, image_np, 'Real Image')
    illustration(fig, 2, fake_np, 'Fake Image')
    plt.show()


if __name__ == '__main__':
    try:
        date = datetime.now()
        logger.info('Experiment: {}'.format(
            date.strftime('%m/%d/%Y, %H:%M:%S')))
        logger.info('Device: {}'.format(constants.DEVICE))
        logger.info('Model: {}'.format(train_test_f.model_name))
        logger.info('Model summury: {}'.format(
            summary(train_test_f.model, (3, 32, 32))))
        logger.info('Model detail: {}'.format(train_test_f.model.__repr__()))
        logger.info('Loss: {}'.format(train_test_f.loss))
        logger.info('Batch size: {}'.format(constants.BATCH_SIZE))
        logger.info('Learning rate: {}'.format(constants.LR))
        comment = input('Comment: ')
        logger.info('Comment: {}'.format(comment))
        t0 = time()
        test_loss_list = []
        n = 0
        model = train_test_f.model.to(constants.DEVICE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=train_test_f.optimizer,
            mode='min',
            verbose=True
        )
        for epoch in range(constants.EPOCHS):
            train(model, train_loader)
            test_loss = test(model, test_loader)
            scheduler.step(test_loss)
            test_loss_list.append(round(test_loss, 5))
            t1 = (time() - t0) / 60
            logger.info(
                'Epoch: {}, test loss: {:.5f}, time: {:.2f} min'.format(
                    epoch+1, test_loss, t1))
            print('Epoch: {}, test loss: {:.5f}, time: {:.2f} min'.format(
                    epoch+1, test_loss, t1))
            if test_loss <= min(test_loss_list):
                n = 0
                continue
            n += 1
            if n > constants.NO_PROGRESS_EPOCHS:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Learning has been stopped manually')
    finally:
        for image, _ in test_loader:
            fake = model(image.to(constants.DEVICE))
            imshow(image, fake, 'image', 'fake')
            break
        logger.info('----------------')
