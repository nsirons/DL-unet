import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate
import cv2 as cv
import os
from sklearn.metrics import f1_score

import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm # See progress in terminal

from data import download_all_data
from network import Unet
from functions import weighted_map

CUR_DIR = os.path.abspath('')



def preprocess_gt(img):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    mask_global = np.zeros(img.shape)
    for cls in np.unique(img):
        if cls == 0:  # if not a background
            continue
        mask_cls = np.zeros(img.shape)
        mask_cls[img==cls] = 255  # binary image of cell
        dilated = cv.dilate(mask_cls, kernel, iterations=2)
        mask_global += dilated-mask_cls  # add edge to global mask
#         mask += cv.erode(dilated-thresh,kernel2,iterations=1) 
#         mask +=  cv.morphologyEx(dilated-thresh, cv.MORPH_OPEN, kernel)
    
    gt = img - mask_global # edges of cells will be background on the new ground truth
    gt[gt<0] = 0  # clipping
    
    return gt, mask_global

# Download data
download_all_data()

# Specify which dataset to analyse
# DATASET = 'PhC-C2DH-U373'
# DATASET = 'ISBI2012'
DATASET = 'DIC-C2DH-HeLa'

#MODE = "TRAINING"
MODE = "TESTING"
if MODE == "TESTING":
    model_path = "unet_weight_save_5000.pth"
else:
    model_path = None

target_path = os.path.join(CUR_DIR, "data", f"{DATASET}-training", "01_GT", "SEG")
training_path = os.path.join(CUR_DIR, "data", f"{DATASET}-training", "01")
target = os.listdir(target_path)

def elastic_transform(images, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       source: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = images[0].shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # TODO: change to bicubic interpolation
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        
    return (map_coordinates(image, indices, order=1).reshape(shape) for image in images)

def mirror_transform(image):
    n = len(image)
    pad = (572 - n) // 2
    new_image = np.zeros((572, 572))
    new_image[pad:pad+n, pad: pad+n] = image
    new_image[:pad, pad:pad + n] = image[pad:0:-1, 0:n]  # left side
    new_image[:pad, :pad] = new_image[0:pad, pad+pad:pad:-1]  # top left
    new_image[:pad, pad+n:] = new_image[0:pad, n+pad-1:n-1:-1]  # bot left
    new_image[n+pad:, pad:pad + n] = image[n:n-pad-1:-1, 0:n]  # right side
    new_image[n+pad:, :pad] = new_image[n+pad:, pad+pad:pad:-1]  # top right
    new_image[n+pad:, n+pad:] = new_image[n+pad:, n+pad-1:n-1:-1]  # bot right
    new_image[pad:n+pad, 0:pad] = image[:, pad:0:-1] # top side
    new_image[pad:n+pad, n+pad:] = image[:, n-1:n-1-pad:-1]  # bot side
    return new_image

class ImageDataset(Dataset):
    
    def __init__(self, root_dir, alpha=3, sigma=10, transform=True):
        self.root_dir = root_dir
        self.alpha = alpha
        self.sigma = sigma

        self.image = []
        self.target = []
        self.transform = transform

        n = len(os.listdir(root_dir)) // 3
        for i in range(1, n+1):
            image_folder = os.path.join(root_dir, f"0{i}")
            target_folder = os.path.join(os.path.join(root_dir, f"0{i}_GT", "SEG"))
            image_names = [filename.replace('man_seg', 't') for filename in os.listdir(target_folder)]
            self.image.extend(cv.imread(os.path.join(image_folder, image_name),-1) for image_name in image_names)
            for filename in os.listdir(target_folder):
                img = cv.imread(os.path.join(target_folder, filename), -1)
                gt, _ = preprocess_gt(img)
                _, gt_bin = cv.threshold(gt, 0, 255, cv.THRESH_BINARY)
                self.target.append(gt_bin)


    def __len__(self):
        if self.transform:
            return len(self.image)
        return 4*len(self.image)  # hardcoded

    def __getitem__(self, idx):
        if self.transform:
            # get images
            image = self.image[idx]
            target = self.target[idx]
            # random crop
            x = np.random.randint(0, 512-388)
            y = np.random.randint(0, 512-388)
            # mirror border
            image = mirror_transform(image[x:x+388, y:y+388])
            target = mirror_transform(target[x:x+388, y:y+388])
            # perform same elastic transformation 
            inp, gt = elastic_transform((image, target), alpha=self.alpha, sigma=self.sigma)
        else:
            # TODO: hardcoded
            i = idx // 4
            x = (512-388)*(idx % 2)
            y = (512-388)*((idx-4*i) // 2)
            inp = mirror_transform(self.image[i][x:x+388, y:y+388])
            gt = mirror_transform(self.target[i][x:x+388, y:y+388])
        
        gt = gt[92:388+92, 92:388+92]  # crop gt
        _, gt = cv.threshold(gt, 127, 255, cv.THRESH_BINARY)
        gt = gt / 255  # normalize to [0 1]
        inp = (inp - np.min(inp))/np.ptp(inp)  # normalize to [0 1]
        return transforms.ToTensor()(inp.astype('float32')), \
               transforms.ToTensor()(gt).long()


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


def training(unet, train_loader, epochs, batch_size, device):
    learning_rate=0.0001
    momentum=0.99

    # Definition of loss function and optimizer
    optimizer = optim.SGD(unet.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):
        m = 0
        for batch in train_loader: # get batch
            images, labels = batch
            optimizer.zero_grad()
            preds = unet(images.to(device)) # pass batch to the unet
            ll = np.zeros((1,2,388,388))
            ll[:,0,:,:] = 1 - labels  # background
            ll[:,1,:,:] = labels  # cell
            ll = torch.from_numpy(ll).to(device)
            weight_maps = weighted_map(labels, batch_size).to(device)
            criterion = nn.BCEWithLogitsLoss(weight=weight_maps)
            loss = criterion(preds, ll)
            loss.backward() # compute the gradients using backprop
            optimizer.step() # update the weights
            m += loss
        print("Epoch:", epoch, m / len(train_loader))
    
        if epoch % 500 == 0:
            PATH = './unet_weight_save_{}.pth'.format(epoch)
            torch.save(unet.state_dict(), PATH)
        #if epoch % 5 == 0:
        #    # Save after training
        #    PATH = './unet_weight_save_5.pth'
        #    torch.save(unet.state_dict(), PATH)
        
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

torch.cuda.empty_cache()
unet = Unet().to(device)
if MODE == "TRAINING":
    training(unet, train_loader,5001,train_loader.batch_size, device)
elif MODE == "TESTING":
    if 'output' not in os.listdir(os.path.join(CUR_DIR, 'data')):
        os.mkdir(os.path.join(CUR_DIR, 'data', 'output'))

    unet.load_state_dict(torch.load(model_path))
    pixel_error = 0
    for i, batch in enumerate(test_clean_loader):
        if i % 4 == 0:
            final_pred = np.zeros((512,512))
            final_overlap_times = np.zeros((512,512))
            fig = plt.figure(figsize=(16, 9))

        ax1 = fig.add_subplot(5,3,3*(i%4)+1)
        ax2 = fig.add_subplot(5,3,3*(i%4)+2)
        ax3 = fig.add_subplot(5,3,3*(i%4)+3)

        x = (512-388)*(i % 2)
        y = (512-388)*((i-4*(i//4)) // 2)
        images, labels = batch
        
        ax1.imshow(images.numpy()[:,:,92:92+388, 92:92+388].reshape((388,388)), cmap='gray')
        ax2.imshow(labels.numpy().reshape((388,388)), cmap='gray')

        preds = unet(images.to(device)).argmax(dim=1).to("cpu").detach().numpy().reshape((388,388))
        
        final_pred[x:x+388, y:y+388] += preds
        final_overlap_times[x:x+388, y:y+388] += np.ones((388, 388))
        
        ax3.imshow(preds, cmap='gray')

        if i % 4 == 3:
            final_pred = np.multiply(final_pred, 1/final_overlap_times)
            final_pred = (final_pred >= 0.5).astype('uint')
            axf = fig.add_subplot(5,3,13)
            axf.imshow(final_pred, cmap='gray')
            fig.tight_layout()
            fig.savefig(os.path.join(CUR_DIR, 'data', 'output', f'final_weight_{i//4}.png'))
        
        labels = labels.reshape((388*388,1))
        preds = preds.reshape((388*388,1))
        
        pixel_error_patch = 1-f1_score(labels, preds)
        pixel_error += pixel_error_patch
        print(pixel_error_patch)
    print(f"Average pixel error over all patches: {pixel_error/len(test_clean_loader):0.4f}")

