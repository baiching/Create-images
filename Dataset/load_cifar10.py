import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import numpy as np

import torchvision
import torchvision.transforms as T

from .chunkSampler import ChunkSampler

def load_cifar():

    NUM_TRAIN = 50000
    NUM_VAL = 10000

    NOISE_DIM = 100
    batch_size = 128

    cifar_train = dset.CIFAR10('./Cifar', train=True, download=True,
                           transform=T.ToTensor())

    loader_train = DataLoader(cifar_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

    cifar_val = dset.CIFAR10('./Cifar', train=True, download=True,
                           transform=T.ToTensor())
    loader_val = DataLoader(cifar_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
    
    return loader_train, loader_val

