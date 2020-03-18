""" Training scheme:

1) Definition of the dataloaders for training and testing sets
2) Initialization of the weights (Gaussian, mean = 0 (?), std = sqrt(2/N))
3) Computation of the weight map (including computation of d1 and d2)    #I think this part has to be removed (the weight map is used in the main for the gt)
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

# Initialization of the network
unet = network.Unet()

#Check for available GPUs
use_gpu = torch.cuda.is_available()

if use_gpu:
    unet = unet.cuda()

# Training 
num_epochs = 20
batch_size = 4

def training(unet, device, transformed_dataset, epochs=num_epochs, batch_size=batch_size, learning_rate=0.001, momentum=0.99, val_per=0.3):

    # Definition of loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(unet.parameters(), lr=learning_rate, momentum=momentum)

    # Creation of the batches
    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in train_loader: # get batch
            images, labels = batch

            preds = unet(images) # pass batch to the unet
            loss = criterion.(preds, labels) #c ompute the loss with the chosen criterion

            optimizer.zero_grad() # set the gradient to zero (needed for the loop structure used to avoid the new gradients computed are added to the old ones)
            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds,labels)

        print("Epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)


    # previous unfinished code:
    
    # Adapt this part to the actual dataset
    dataset = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])

    for epoch in range(num_epochs):

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size
        )

        batch = next(iter(data_loader))

        images, labels = batch

        w = functions.weighted_map(gt)