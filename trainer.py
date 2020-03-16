""" Training scheme:

1) Definition of the dataloaders for training and testing sets
2) Initialization of the weights (Gaussian, mean = 0 (?), std = sqrt(2/N))
3) Computation of the weight map (including computation of d1 and d2)
4) Define CrossEntopyLoss (weighted loss)
5) Optimizer (SGD with momentum = 0.99, LR?)
6) Define number of epochs, batch_size, maybe variable LR?
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import network
import functions
import data

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm # See progress in terminal


# Path definition:

path_images = "/Users/pere/opt/anaconda3/envs/DL_Delft/reproducibility/unet/images"
path_labels = "/Users/pere/opt/anaconda3/envs/DL_Delft/reproducibility/unet/labels"

use_gpu = torch.cuda.is_available()

unet = network.Unet()

if use_gpu:
    unet = unet.cuda()

num_epochs = 20
batch_size = 4

def training(unet, device, epochs=num_epochs, batch_size=batch_size, learning_rate=0.001, momentum=0.99, val_per=0.3):

    # Adapt this part to the actual dataset
    dataset = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unet.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size
        )

        batch = next(iter(data_loader))

        images, labels = batch

        w = functions.weighted_map(gt)