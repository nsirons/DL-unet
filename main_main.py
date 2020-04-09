import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import argparse

import torch
from torch.utils.data import DataLoader

# from sklearn.metrics import f1_score

from data import download_all_data, ImageDataset, ImageDataset_test
from network import Unet
from trainer import training, training_all

##############################

# Arguments in command line
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, help='training or testing. Specify by writing '
                                                   '::TRAINING:: or ::TESTING::.', required=True)

parser.add_argument('-d', '--dataset', type=str, help='specify dataset for training or testing.'
                                                      'Options: ::DIC-C2DH-HeLa::, ::ISBI2012::, '
                                                      'or ::PhC-C2DH-U373::.', required=True)

parser.add_argument('-f', '--folds', type=int, help='number of folds for the cross validation'
                                                    'If we do not input FOLDS, we will train with '
                                                    'the whole training set.', required=False)

args = parser.parse_args()

MODE    = args.mode
DATASET = args.dataset
FOLDS   = args.folds

##############################

# Curent directory
CUR_DIR = os.path.abspath('')
# Choose dataset
root_dir = os.path.join(CUR_DIR, 'data', f'{DATASET}-training')

##############################

# Download data
download_all_data()

##############################

# Parameters:
if FOLDS is None: # Train with the full training set
    val_per = 0.0
elif FOLDS > 5:
    print('Number of folds is bigger than 5')
else:
    val_per = 0.2

tr_per  = 1.0 - val_per
batch_size = 10

##############################

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on the GPU')

else:
    device = torch.device('cpu')
    print('Running on the CPU')

##############################

if MODE == 'TRAINING':
    # Init dataset
    train_dataset = ImageDataset(root_dir, alpha=200, sigma=10)   # For training + validation

    # Determine number of samples in training and validation
    samp_tr  = int(np.round(tr_per  * len(train_dataset)))
    samp_val = int(np.round(val_per * len(train_dataset)))

    # Round numbers so that we do not exceed total number of samples
    while samp_tr + samp_val > len(train_dataset): samp_val += -1

    # Generate an order vector to shuffle the samples before each fold for the cross validation  
    order = np.arange(len(train_dataset))
    np.random.shuffle(order)
    
    if FOLDS is None: # Training with all available samples
        all_dir = os.path.join(CUR_DIR, 'models', f'{DATASET}', 'all')
        maybe_mkdir_p(fold_dir)

        # Suffle and load the training set
        train_loader = DataLoader(train_set, batch_size=batch_size , shuffle=True, num_workers=1) # num_workers?

        torch.cuda.empty_cache()
        unet = Unet().to(device)

        training_all(unet, train_loader, epochs=5000, batch_size=batch_size, device=device, all_dir=all_dir)

    else:
        for fold in range(FOLDS): # Cross validation
            # Make directory where we save model and data for each fold
            fold_dir = os.path.join(CUR_DIR, 'models', f'{DATASET}', f'fold{fold}')
            maybe_mkdir_p(fold_dir)

            # Order the training set (first time shuffles, the rest is determined by the shift in order below)
            train_dataset = [train_dataset[i] for i in order]

            # Divide full train_dataset between training and validation
            train_set   = train_dataset[0: samp_tr]
            val_set     = train_dataset[samp_tr: -1]

            train_loader = DataLoader(train_set, batch_size=batch_size , shuffle=True, num_workers=1) # num_workers?
            val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True)

            # Shift values in order for next fold of cross validation (a shift of samp_val)
            order = np.append(order[samp_val:], order[0:samp_val])
                
            torch.cuda.empty_cache()
            unet = Unet().to(device)    

            training(unet, train_loader, val_loader, epochs=5000, batch_size=batch_size, device=device, fold_dir=fold_dir)

elif MODE == 'TESTING': 
    model_path = 'unet_weight_save_5000.pth'

    test_dataset  = ImageDataset_test(root_dir)                   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fold_dir = os.path.join(CUR_DIR, 'models', f'{DATASET}', f'fold{fold}')
    maybe_mkdir_p(os.path.join(root_dir))

    if 'output' not in os.listdir(os.path.join(CUR_DIR, 'data')):
        os.mkdir(os.path.join(CUR_DIR, 'data', 'output'))

    unet.load_state_dict(torch.load(model_path))

    pixel_error = 0

    for i, batch in enumerate(test_loader):

