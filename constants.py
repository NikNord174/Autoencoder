import torch


BATCH_SIZE = 64
NUM_WORKERS = 2
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('mps')
epochs = 3
running_loss = 0.0
lr = 0.001
eps = 1e-3
