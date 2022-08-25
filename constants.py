import torch

# device
'''if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')'''
DEVICE = torch.device('cpu')

# models parameters
POOL_MODE = 'bilinear'
ReLU_FACTOR = 0.2

# learning parameters
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 20
LR = 1e-4
NO_PROGRESS_EPOCHS = 5
