import torch


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
BATCH_SIZE = 64
NUM_WORKERS = 2
MODE = 'bilinear'
EPOCHS = 20
LR = 0.001
NO_PROGRESS_EPOCHS = 5
