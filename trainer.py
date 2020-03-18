""" Training scheme:

1) Definition of the dataloaders for training and testing sets
2) Initialization of the weights (Gaussian, mean = 0 (?), std = sqrt(2/N))
3) Computation of the weight map (including computation of d1 and d2)    #I think this part has to be removed (the weight map is used in the main for the gt)
4) Define CrossEntopyLoss (weighted loss)
5) Optimizer (SGD with momentum = 0.99, LR?)
6) Define number of epochs, batch_size, maybe variable LR?

Considerations:
    Preprocessing?
    Data augmentation?
    Do we have to do something regarding GPUs? (load tensors or sth?)
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import network
import functions
import data

# from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm # See progress in terminal

# Initialization of the network
unet = network.Unet()

#Check for available GPUs
use_gpu = torch.cuda.is_available()

if use_gpu:
    unet = unet.cuda()

def training(unet, train_loader, epochs, batch_size):
    
    learning_rate=0.001
    momentum=0.99

    # Definition of loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unet.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(num_epochs):

        for batch in train_loader: # get batch

            images, labels = batch
            print("BATCH START")

            preds = unet(images) # pass batch to the unet

            print("BATCH NETWORK")

            weight_maps = functions.weighted_map(labels, batch_size)

            print(preds.shape, labels.shape, weight_maps.shape)

            loss = criterion(preds, labels) # compute the loss with the chosen criterion

            print("BATCH loss")

            optimizer.zero_grad() # set the gradient to zero (needed for the loop structure used to avoid the new gradients computed are added to the old ones)
            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights
            print("BATCH END")

        print("Epoch:", epoch)