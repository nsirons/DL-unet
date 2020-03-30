import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions import weighted_map

# from torch.utils.tensorboard import SummaryWriter 
# from tqdm import tqdm # See progress in terminal

def training(unet, train_loader, epochs, batch_size, device):

    learning_rate=0.0001
    momentum=0.99

    # Definition of loss function and optimizer

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = custom_loss()

    optimizer = optim.SGD(unet.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):
        
        print(' ')
        print('Epoch:', epoch)

        total_loss = 0

        batch_id = 0

        for batch in train_loader: # get batch

            optimizer.zero_grad()

            print(' BATCH START', batch_id)

            print(' Batch to network:')
            images, labels = batch
            preds = unet(images.to(device)) # pass batch to the unet
            print(' done')

            ll = torch.empty_like(preds)
            ll[:,0,:,:] = 1 - labels  # background
            ll[:,1,:,:] = labels  # cell
            ll = ll.to(device)

            weight_maps = weighted_map(labels, batch_size).to(device)
            criterion = criterion(weight=weight_maps)
            loss = criterion(preds, ll)
            print(' Batch loss:', loss)

            print(' Updating weights...')
            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights
            print(' done')

            total_loss += loss

            batch_id =+ 1

        print('Total loss:', total_loss / len(train_loader))
        print(' ')
    
        # Save model every 500 epochs
        if epoch % 500 == 0:
            PATH = './unet_weight_save_{}.pth'.format(epoch)
            torch.save(unet.state_dict(), PATH)
            
def custom_loss(pred, label, weights):
    batch_size, c, h, w = pred.shape
    logp = -F.log_softmax(pred)  # added - sign

    # Gather log probabilities with respect to target
    logp = logp.gather(1, label.view(batch_size, 1, h, w))
    # plt.imshow(logp.detach().numpy()[0,0,:,:], cmap='gray')
    # plt.colorbar()
    # plt.show()
    # print(logp.shape)
    # Multiply with weights
    weighted_logp = (logp * weights).view(batch_size, -1)

    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
    
    # Average over mini-batch
    weighted_loss = weighted_loss.mean()
    return weighted_loss