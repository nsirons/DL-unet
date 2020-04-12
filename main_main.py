''' Main script 

We use this script to perform training and/or testing for the Unet
described in Ronneberger et al. (2015)[1].

This project is framed in the reproducibility assignement from the 
Deep Learning course (CS4240) at TU Delft (Spring 2020).

Authors (alph): Canals, P., Monguzzi, A., & Sirons, N.

GitHub: https://github.com/nsirons/DL-unet

[1] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional 
networks for biomedical image segmentation. Lecture Notes in Computer Science 
(Including Subseries Lecture Notes in Artificial Intelligence and Lecture 
Notes in Bioinformatics), 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28

------------------------------

To run the code, you should be in the same dir as the script and run the following 
line in the terminal command line:

>> python3 main_main.py -m MODE -d DATASET -f FOLDS -n NETWORK

Arguments:
    - MODE: either 'TRAINING' or 'TESTING'. Required.
    - DATASET: you can choose either of the three datasets present in [1], i.e, 
               'DIC-C2DH-HeLa', 'ISBI2012' or 'PhC-C2DH-U373'. Required.
    - FOLDS: number of folds for the cross validation. Integer number. Not 
             required, but not inputing it in TRAINING mode means training
             with the whole dataset.
    - NETWORK: path to the model that we want to test in TESTING mode. 
               it assumes that you are in CUR_DIR. Required in TESTING mode.
    - SEED: random seed to determine how training/validation images are managed.
            Not required.
    - START_FROM: in case one wants to continue training from last model saved, 
                  input an int =~ 0 (cross validation) and input -1 for training_all.
    - SKIP_FOLD: if we want to skip any folds from the same random seed, we shall
                 use this argument. See help for example. Not required.

'''

import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
import argparse

import torch
from torch.utils.data import DataLoader

from data import download_all_data, ImageDataset, ImageDataset_test
from network import Unet
from trainer import training
from tester import testing

##############################

# Arguments in command line
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', type=str, help='training or testing. Specify by writing '
                                                   '::TRAINING:: or ::TESTING::.', required=True)

parser.add_argument('-d', '--dataset', type=str, help='specify dataset for training or testing. '
                                                      'Options: ::DIC-C2DH-HeLa::, ::ISBI2012::, '
                                                      'or ::PhC-C2DH-U373::.', required=True)

parser.add_argument('-f', '--folds', type=int, help='number of folds for the cross validation. '
                                                    'If we do not input FOLDS, we will train with '
                                                    'the whole training set.', required=False)

parser.add_argument('-n', '--network', type=str, help='path to the model that we want to use for '
                                                      'TESTING.', required=False)

parser.add_argument('-s', '--seed', type=int, help='random seed for the dataset ordering. Not '
                                                   'required.', required=False)

parser.add_argument('-sf', '--start_from', type=int, help='continue training from last saved model. '
                                                          'Options: ::-1::, - continue training full model. '
                                                          'Options: ::N::, - start from Nth fold.', required=False)

parser.add_argument('-sk', '--skip_fold', type=int, help='it skips folds below number specified. '
                                                         'E.g. if --skip_fold is set to ::1::, it will '
                                                         'skip the first fold (fold 0). Not required.', required=False)

args = parser.parse_args()

MODE       = args.mode
DATASET    = args.dataset
FOLDS      = args.folds
NETWORK    = args.network
SEED       = args.seed
START_FROM = args.start_from
SKIP_FOLD  = args.skip_fold

print(' '                                                                                           )
print('Reproduced U-net (Ronneberger et al. 2015), by Canals, P., Monguzzi, A., & Sirons, N. (2020)')
print(' '                                                                                           )
print('------------------------------'                                                              )
print(' '                                                                                           )
print('Mode:   ', MODE                                                                              )
print('Dataset:', DATASET                                                                           )
print(' '                                                                                           )

##############################

# Curent directory
CUR_DIR = os.path.abspath('')
# Choose dataset
root_dir = os.path.join(CUR_DIR, 'data', f'{DATASET}-training')

##############################

# Download data (if it has not been downoalded already)
download_all_data()

##############################

# Parameters:
if FOLDS is None: # Train with the full training set
    print('Training with all available training data')
    val_per = 0.0
elif FOLDS > 5:
    raise ValueError('Input a FOLDS value below 5')
else:
    print('Folds (cross validation):', FOLDS)
    val_per = 0.2
    print('Training:validation = {}:{}'.format(int(100*(1-val_per)), int(100*val_per)))

if SEED is None: SEED = 0

if SKIP_FOLD is None: SKIP_FOLD = 0

tr_per  = 1.0 - val_per
batch_size = 2
epochs = 500
print('Batch size: ', batch_size)
print('                        ')
print('Seed:', SEED             )

##############################

# Special treatment of training data in the case of the ISBI2012 dataset
if DATASET == 'ISBI2012': 
    ISBI2012 = True
else: 
    ISBI2012 = False

##############################

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(' ')
    print('Running on the GPU')
    print(' ')

else:
    device = torch.device('cpu')
    print(' ')
    print('Running on the CPU')
    print(' ')

##############################

if MODE == 'TRAINING':
    # Init dataset
    print('Initializing dataset...')
    print(' ')
    train_dataset = ImageDataset(root_dir, alpha=200, sigma=10, ISBI2012=ISBI2012)   # For training + validation (in case of FOLDS = None, only for training)

    # Determine number of samples in training and validation
    samp_tr  = int(np.round(tr_per  * len(train_dataset)))
    samp_val = int(np.round(val_per * len(train_dataset)))

    # Round numbers so that we do not exceed total number of samples
    while samp_tr + samp_val > len(train_dataset): samp_val += -1

    # Generate an order vector to shuffle the samples before each fold for the cross validation  
    np.random.seed(SEED)
    order = np.arange(len(train_dataset))
    np.random.shuffle(order)
    
    if FOLDS is None: # Training with all available samples
        # Make directory where we save model and data
        all_dir = os.path.join(CUR_DIR, 'models', f'{DATASET}', 'all')
        maybe_mkdir_p(all_dir)

        val_dataset = ImageDataset_test(root_dir, ISBI2012=ISBI2012)

        # Suffle and load the training set
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)

        torch.cuda.empty_cache()
        unet = Unet().to(device)
        if START_FROM == -1:  # load latest model
            epoch_id = max([int(name.replace('unet_weight_save_', '').replace('.pth', '')) for name in os.listdir(os.path.join(all_dir, 'models'))])
            PATH = os.path.join(all_dir, 'models', 'unet_weight_save_{}.pth'.format(epoch_id))
            unet.load_state_dict(torch.load(PATH))

        print('Number of images used for training:', len(train_dataset))
        print('                                                       ')
        print('Starting training'                                      )
        print('                                                       ')

        training(unet, train_loader, val_loader, epochs=epochs, batch_size=batch_size, device=device, fold_dir=all_dir, dataset=DATASET)

    else:
        for fold in range(FOLDS): # Cross validation
            if  fold < SKIP_FOLD:
                print('Skipping fold', fold)
                print('                   ')
            else:
                print('Starting training: fold', fold)
                # Make directory where we save model and data for each fold
                fold_dir = os.path.join(CUR_DIR, 'models', f'{DATASET}', f'fold{fold}')
                maybe_mkdir_p(fold_dir)

                # Order the training set (first time shuffles, the rest is determined by the shift in order below)
                train_dataset = [train_dataset[idx] for idx in order]

                # Divide full train_dataset between training and validation
                train_set   = train_dataset[0: samp_tr]
                val_set     = train_dataset[samp_tr: -1]

                # If testing on CPU use these:
                # train_set   = train_dataset[0: 2]
                # val_set     = train_dataset[2: 3]

                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) 
                val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=True)

                # Shift values in order for next fold of cross validation (a shift of samp_val)
                order = np.append(order[samp_val:], order[0:samp_val])
                    
                torch.cuda.empty_cache()
                unet = Unet().to(device)
                if START_FROM is not None:  # load latest model
                    # find latest model epoch id
                    epoch_id = max([int(name.replace('unet_weight_save_', '').replace('.pth', '')) for name in os.listdir(os.path.join(fold_dir, 'models'))])
                    print('Starting from Epoch', epoch_id)
                    PATH = os.path.join(fold_dir, 'models', 'unet_weight_save_{}.pth'.format(epoch_id))
                    unet.load_state_dict(torch.load(PATH))

                print('Number of images used for training  :', len(train_set))
                print('Number of images used for validation:', len(val_set)  )  
                print('                                                     ')
                print('Starting training'                                    )
                print('                                                     ')

                training(unet, train_loader, val_loader, epochs=epochs, batch_size=batch_size, device=device, fold_dir=fold_dir, dataset=DATASET)

elif MODE == 'TESTING': 
    # Get model path to test
    if NETWORK is not None:
        model_path = os.path.join(CUR_DIR, NETWORK)
    else:
        raise ValueError('Input a network path when calling the script')

    # Get test data
    test_dataset = ImageDataset_test(root_dir, ISBI2012=ISBI2012)   
    test_dataset = [test_dataset[idx] for idx in range(len(test_dataset))]               
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Make directory for test outputs (assumes .pth format)
    output_dir = model_path[0:len(model_path)-4] + '_test'
    maybe_mkdir_p(output_dir)

    torch.cuda.empty_cache()
    unet = Unet().to(device)
    unet.load_state_dict(torch.load(model_path))

    print('Number of images used for testing:', len(test_dataset))
    print('                                                     ')
    print('Starting testing'                                     )
    print('                                                     ')

    testing(unet, test_loader, batch_size=1, device=device, output_dir=output_dir)