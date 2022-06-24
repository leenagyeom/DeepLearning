import os

import torch
import torch.nn as nn


# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyper parameters
batch_size = 16
nc = 2
num_epoch = 10
val_every = 1
lr = 0.01
criterion = nn.CrossEntropyLoss().to(device)

# path
data_path = "./dataset/"
save_path = "./weights"
os.makedirs(save_path, exist_ok=True)
