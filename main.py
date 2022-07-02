import os
from time import time
from datetime import datetime
import torch

import constants
from models import (Autoencoder, Autoencoder_ConvTranspose,
                    Autoencoder_Upsampling)
from train_test_f import criterion, train, test

from data_import import train_loader, test_loader

model = Autoencoder_Upsampling().to(constants.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=constants.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

t0 = time()
test_loss_list = []


def results_recording():
    parameters['epochs:'] = epoch + 1
    parameters['losses:'] = test_loss_list
    parameters['time:'] = '{:.2f}'.format(t1)

    filename = 'experimental_results.txt'
    if not os.path.exists(filename):
        open(filename, 'a').close()
    filehandler = open(filename, 'a')
    for key, value in parameters.items():
        filehandler.write('{} {}\n'.format(str(key), str(value)))
    filehandler.write('-------------------------\n')
    print('Finish!')


if __name__ == '__main__':
    try:
        date = datetime.now()
        comment = (
            "Changed Upsample mode to 'nearest' and stop criterium to 1e-3"
        )
        parameters = {
            'Experiment:': date.strftime('%m/%d/%Y, %H:%M:%S'),
            'model': model.__class__.__name__,
            'model detail:': model,
            'loss function': criterion.__name__,
            'Comments:': comment,
        }
        for epoch in range(constants.epochs):
            loss = 0.0
            train(model, train_loader, optimizer, constants.DEVICE)
            test_loss = test(model, test_loader, constants.DEVICE)
            test_loss_list.append(round(test_loss, 5))
            t1 = (time() - t0) / 60
            print(
                'Epoch: {}, test loss: {:.5f}, time: {:.2f} min'.format(
                    epoch+1, test_loss, t1))
            if epoch > 2:
                if (max(test_loss_list[-5:]) - min(test_loss_list[-5:])
                        > constants.eps):
                    continue
                else:
                    break
        results_recording()
    except KeyboardInterrupt:
        results_recording()
