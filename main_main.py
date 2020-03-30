import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate
from sklearn.metrics import f1_score

import numpy as np
import cv2 as cv
import os

import torchvision
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# from tqdm import tqdm # See progress in terminal
# from torch.utils.tensorboard import SummaryWriter 

from data import download_all_data, preprocess_gt, mirror_transform, elastic_transform, ImageDataset
from network import Unet
from functions import weighted_map
from trainer import custom_loss, training

CUR_DIR = os.path.abspath('')

# Download data
download_all_data()

# Specify which dataset to analyse // Argument when calling the code + readme description?
# DATASET = 'PhC-C2DH-U373'
# DATASET = 'ISBI2012'
DATASET = 'DIC-C2DH-HeLa'

# Specify which mode // Argument when calling the code?
# MODE = 'TRAINING'
MODE = 'TESTING'

if MODE == 'TESTING':
    model_path = 'unet_weight_save_5000.pth'
else:
    model_path = None


target_path = os.path.join(CUR_DIR, 'data', f'{DATASET}-training', '01_GT', 'SEG')
training_path = os.path.join(CUR_DIR, 'data', f'{DATASET}-training', '01')
target = os.listdir(target_path)

# init ImageDataset
root_dir = os.path.join(CUR_DIR, "data", "DIC-C2DH-HeLa-training")
transformed_dataset = ImageDataset(root_dir, alpha=200, sigma=10)
clean_dataset = ImageDataset(root_dir, transform=False)

val_per = 0.3
batch_size = 1 

train_size = int(len(transformed_dataset)*(1-val_per))
train_set = torch.utils.data.Subset(transformed_dataset, range(train_size))  # from 0 ... train_size
test_train_transform_set = torch.utils.data.Subset(transformed_dataset, range(train_size, len(transformed_dataset)))  # from train_size ... end
test_clean_set = torch.utils.data.Subset(clean_dataset, range(4*train_size, len(clean_dataset)))  # TODO: 4 hardcoded

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_train_transform_loader = DataLoader(test_train_transform_set, batch_size=batch_size, shuffle=False)
test_clean_loader = DataLoader(test_clean_set, batch_size=1, shuffle=False)
        
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on the GPU')

else:
    device = torch.device('cpu')
    print('Running on the CPU')


torch.cuda.empty_cache()
unet = Unet().to(device)


if MODE == 'TRAINING':
    training(unet, train_loader,5001,train_loader.batch_size, device)

elif MODE == 'TESTING': # Hardcoded, cleaning needed + prints
    if 'output' not in os.listdir(os.path.join(CUR_DIR, 'data')):
        os.mkdir(os.path.join(CUR_DIR, 'data', 'output'))

    unet.load_state_dict(torch.load(model_path))
    pixel_error = 0
    for i, batch in enumerate(test_clean_loader):
        # if i % 4 == 0:
        #     final_pred = np.zeros((512,512))
        #     final_overlap_times = np.zeros((512,512))
        #     fig = plt.figure(figsize=(16, 9))

        # ax1 = fig.add_subplot(5,3,3*(i%4)+1)
        # ax2 = fig.add_subplot(5,3,3*(i%4)+2)
        # ax3 = fig.add_subplot(5,3,3*(i%4)+3)

        x = (512-388)*(i % 2)
        y = (512-388)*((i-4*(i//4)) // 2)
        images, labels = batch
        
        # ax1.imshow(images.numpy()[:,:,92:92+388, 92:92+388].reshape((388,388)), cmap='gray')
        # ax2.imshow(labels.numpy().reshape((388,388)), cmap='gray')

        preds = unet(images.to(device)).argmax(dim=1).to('cpu').detach().numpy().reshape((388,388))
        
        final_pred[x:x+388, y:y+388] += preds
        final_overlap_times[x:x+388, y:y+388] += np.ones((388, 388))
        
        # ax3.imshow(preds, cmap='gray')

        # if i % 4 == 3:
        #     final_pred = np.multiply(final_pred, 1/final_overlap_times)
        #     final_pred = (final_pred >= 0.5).astype('uint')
        #     axf = fig.add_subplot(5,3,13)
        #     axf.imshow(final_pred, cmap='gray')
        #     fig.tight_layout()
        #     fig.savefig(os.path.join(CUR_DIR, 'data', 'output', f'final_weight_{i//4}.png'))

        
        labels = labels.reshape((388*388,1))
        preds = preds.reshape((388*388,1))
        
        pixel_error_patch = 1-f1_score(labels, preds) 
        pixel_error += pixel_error_patch
        print('Pixer error (patch):',pixel_error_patch)

    print(f'Average pixel error over all patches: {pixel_error/len(test_clean_loader):0.4f}')

