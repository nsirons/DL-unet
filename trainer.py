import numpy as np
import os

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functions import weighted_map, evaluation_metrics

from time import time

def training(unet, train_loader, val_loader, epochs, batch_size, device, fold_dir, dataset):

    # Set goals for training to end
    if dataset is 'DIC-C2DH-HeLa':
        when_to_stop = 0
        goal = 0.7756 # IoU value from table 2 in Ronneberger et al. (2015)
    elif dataset is 'ISBI2012':
        when_to_stop = 1
        goal = 0.0582 # PE value from table 1 in Ronneberger et al. (2015)
    elif dataset is 'PhC-C2DH-U373':
        when_to_stop = 2
        goal = 0.9203 # IoU value from table 2 in Ronneberger et al. (2015)
    else:
        when_to_stop = None

    optimizer = optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-3, eps=1e-6)
    patience = 0

    maybe_mkdir_p(os.path.join(fold_dir, 'progress'))
    maybe_mkdir_p(os.path.join(fold_dir, 'models' ))

    for epoch in range(epochs+1):
        
        print(' ')
        print('Epoch:', epoch)

        start = time()
        total_loss = 0
        previous_loss = 0
        start_eval_train = 0
        start_eval_val = 0

        for batch in train_loader:

            optimizer.zero_grad()

            images, labels = batch

            preds = unet(images.to(device)) # pass batch to the unet

            pad = int((preds.shape[-1] - labels.shape[-1]) / 2)
            preds = preds[:, :, pad:labels.shape[-1]+pad, pad:labels.shape[-1]+pad]

            ll = torch.empty_like(preds)
            ll[:,0,:,:] = 1 - labels[:, 0, :, :] # background
            ll[:,1,:,:] = labels[:, 0, :, :] # cell
            ll = ll.to(device)

            weight_maps = weighted_map(labels, batch_size).to(device)
            criterion = nn.BCEWithLogitsLoss(weight=weight_maps)
            loss = criterion(preds, ll)

            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights

            total_loss += loss

            preds = preds.argmax(dim=1)

            for idx in range(preds.shape[0]):
                if start_eval_train == 0 and idx == 0: # First time in epoch we initialize train_eval
                    train_eval = evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())
                    start_eval_train += 1
                else:
                    np.concatenate((train_eval, evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())), axis=1)

        scheduler.step(total_loss / len(train_loader)) # update the lr

        train_eval_epoch = np.mean(train_eval, axis=1)

        for batch in val_loader:
            
            images, labels = batch

            preds = unet(images.to(device))

            pad = int((preds.shape[-1] - labels.shape[-1]) / 2)
            preds = preds[:, :, pad:labels.shape[-1]+pad, pad:labels.shape[-1]+pad].argmax(dim=1)

            for idx in range(preds.shape[0]):
                if start_eval_val == 0 and idx == 0: # First time in epoch we initialize val_eval
                    val_eval = evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())
                    start_eval_val += 1
                else:
                    np.concatenate((val_eval, evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())), axis=1)

        val_eval_epoch = np.mean(val_eval, axis=1)

        for param_group in optimizer.param_groups: l_rate = param_group['lr']

        loss_epoch = total_loss / len(train_loader)
        loss_diff = abs(previous_loss - loss_epoch)
        if loss_diff < 1e-3: patience += 1
        previous_loss = loss_epoch

        print('Current lr is:        ', l_rate                              )
        print('Patience is:           {}/10'.format(patience)               )
        print('Mean IoU training:    ', "{:.6f}".format(train_eval_epoch[0]))
        print('Mean PE training:     ', "{:.6f}".format(train_eval_epoch[1]))
        print('Mean IoU validation:  ', "{:.6f}".format(val_eval_epoch[0])  )
        print('Mean PE validation:   ', "{:.6f}".format(val_eval_epoch[1])  )
        print('Total training loss:  ', "{:.6f}".format(loss_epoch.item())  )
        print('Epoch duration:       ', "{:.6f}".format(time()-start), 's'  )
        print(' ')

        if patience == 10: patience = 0

        # Save progress (evaluation metrics and loss)
        if epoch == 0:
            train_eval_progress_iou = [train_eval_epoch[0]]
            train_eval_progress_pe  = [train_eval_epoch[1]]
            val_eval_progress_iou   = [val_eval_epoch[0]]
            val_eval_progress_pe    = [val_eval_epoch[1]]
            loss_progress           = [loss_epoch.item()]
        elif epoch > 0:
            train_eval_progress_iou = np.concatenate((train_eval_progress_iou, [train_eval_epoch[0]]))
            train_eval_progress_pe  = np.concatenate((train_eval_progress_pe, [train_eval_epoch[1]]) )
            val_eval_progress_iou   = np.concatenate((val_eval_progress_iou, [val_eval_epoch[0]])    )
            val_eval_progress_pe    = np.concatenate((val_eval_progress_pe, [val_eval_epoch[1]])     )
            loss_progress           = np.append(loss_progress, [loss_epoch.item()])

        np.savetxt(os.path.join(fold_dir, 'progress', 'train_eval_iou.out'), train_eval_progress_iou)
        np.savetxt(os.path.join(fold_dir, 'progress', 'train_eval_pe.out' ), train_eval_progress_pe )
        np.savetxt(os.path.join(fold_dir, 'progress', 'val_eval_iou.out'  ), val_eval_progress_iou  )
        np.savetxt(os.path.join(fold_dir, 'progress', 'val_eval_pe.out'   ), val_eval_progress_pe   )
        np.savetxt(os.path.join(fold_dir, 'progress', 'loss.out'          ), loss_progress          )

        if when_to_stop == 0:
            if val_eval_epoch[0] > goal:
                PATH = os.path.join(fold_dir, 'unet_weight_save_{}.pth'.format(dataset))
                torch.save(unet.state_dict(), PATH)
                print('The goal was reached in epoch {}!'.format(epoch))
                print('Model has been saved:')
                print('     ', PATH)
                break
            continue
        elif when_to_stop == 1:
            if val_eval_epoch[0] > goal:
                PATH = os.path.join(fold_dir, 'unet_weight_save_{}.pth'.format(dataset))
                torch.save(unet.state_dict(), PATH)
                print('The goal was reached in epoch {}!'.format(epoch))
                print('Model has been saved:')
                print('     ', PATH)
                break
            continue
        elif when_to_stop == 2:
            if val_eval_epoch[0] > goal:
                PATH = os.path.join(fold_dir, 'unet_weight_save_{}.pth'.format(dataset))
                torch.save(unet.state_dict(), PATH)
                print('The goal was reached in epoch {}!'.format(epoch))
                print('Model has been saved:')
                print('     ', PATH)
                break
            continue

        # Save model every 500 epochs
        if epoch % 500 == 0:
            PATH = os.path.join(fold_dir, 'models', 'unet_weight_save_{}.pth'.format(epoch))
            torch.save(unet.state_dict(), PATH)
            print('Model has been saved:')
            print('     ', PATH)

    print('Training is finished as epoch {} has been reached'.format(epochs))



def training_all(unet, train_loader, epochs, batch_size, device, all_dir):

    optimizer = optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-3, eps=1e-6)
    patience = 0

    for epoch in range(epochs+1):
        
        print(' ')
        print('Epoch:', epoch)

        start = time()
        total_loss = 0
        previous_loss = 0
        start_eval_train = 0

        maybe_mkdir_p(os.path.join(all_dir, 'progress'))

        for batch in train_loader:

            optimizer.zero_grad()

            images, labels = batch

            preds = unet(images.to(device)) # pass batch to the unet

            pad = int((preds.shape[-1] - labels.shape[-1]) / 2)
            preds = preds[:, :, pad:labels.shape[-1]+pad, pad:labels.shape[-1]+pad]

            ll = torch.empty_like(preds)
            ll[:,0,:,:] = 1 - labels[:, 0, :, :] # background
            ll[:,1,:,:] = labels[:, 0, :, :] # cell
            ll = ll.to(device)

            weight_maps = weighted_map(labels, batch_size).to(device)
            criterion = nn.BCEWithLogitsLoss(weight=weight_maps)
            loss = criterion(preds, ll)

            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights

            total_loss += loss

            preds = preds.argmax(dim=1)

            for idx in range(preds.shape[0]):
                if start_eval_train == 0 and idx == 0: # First time in epoch we initialize train_eval
                    train_eval = evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())
                    start_eval_train += 1
                else:
                    np.concatenate((train_eval, evaluation_metrics(preds[idx, :, :].detach(), labels[idx, 0, :, :].detach())), axis=1)

        scheduler.step(total_loss / len(train_loader)) # update the lr

        train_eval_epoch = np.mean(train_eval, axis=1)

        for param_group in optimizer.param_groups: l_rate = param_group['lr']

        loss_epoch = total_loss / len(train_loader)
        loss_diff = abs(previous_loss - loss_epoch)
        if loss_diff < 1e-3: patience += 1
        previous_loss = loss_epoch

        print('Current lr is:        ', l_rate                              )
        print('Patience is:           {}/10'.format(patience)               )
        print('Mean IoU training:    ', "{:.6f}".format(train_eval_epoch[0]))
        print('Mean PE training:     ', "{:.6f}".format(train_eval_epoch[1]))
        print('Total training loss:  ', "{:.6f}".format(loss_epoch.item())  )
        print('Epoch duration:       ', "{:.6f}".format(time()-start), 's'  )
        print(' ')

        if patience == 10: patience = 0

        # Save progress (evaluation metrics and loss)
        if epoch == 0:
            train_eval_progress_iou = [train_eval_epoch[0]]
            train_eval_progress_pe  = [train_eval_epoch[1]]
            loss_progress           = [loss_epoch.item()]
        elif epoch > 0:
            train_eval_progress_iou = np.concatenate((train_eval_progress_iou, [train_eval_epoch[0]]))
            train_eval_progress_pe  = np.concatenate((train_eval_progress_pe, [train_eval_epoch[1]]) )
            loss_progress           = np.append(loss_progress, [loss_epoch.item()]                   )

        np.savetxt(os.path.join(all_dir, 'progress', 'train_eval_iou.out'), train_eval_progress_iou)
        np.savetxt(os.path.join(all_dir, 'progress', 'train_eval_pe.out' ), train_eval_progress_pe )
        np.savetxt(os.path.join(all_dir, 'progress', 'loss.out'          ), loss_progress          )
    
        # Save model every 500 epochs
        if epoch % 500 == 0:
            PATH = os.path.join(all_dir, 'models', 'unet_weight_save_{}.pth'.format(epoch))
            torch.save(unet.state_dict(), PATH)
            print('Model has been saved:')
            print('     ', PATH)

    print('Training is finished as epoch {} has been reached'.format(epochs))


            
def custom_loss(pred, label, weights):
    batch_size, c, h, w = pred.shape
    logp = -F.log_softmax(pred)  # added - sign

    # Gather log probabilities with respect to target
    logp = logp.gather(1, label.view(batch_size, c, h, w))

    # Multiply with weights
    weighted_logp = (logp * weights).view(batch_size, -1)

    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)
    
    # Average over mini-batch
    weighted_loss = weighted_loss.mean()
    return weighted_loss