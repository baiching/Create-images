import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

from .helper import Flatten, Unflatten, initialize_weights
import numpy as np

def build_dc_classifier(batch_size=128):
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        Unflatten(batch_size, 3, 32, 32),
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        Flatten(),
        nn.Linear(1600, 1024, bias=True),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, 1, bias=True)
    )

def build_dc_generator(noise_dim=100, batch_size=128):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        ###########################
        nn.Linear(in_features=noise_dim, out_features=1024, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        
        nn.Linear(in_features=1024, out_features=8192, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(8192),
        
        Unflatten(batch_size, 128, 8, 8),
        
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )
