import os
import wget
import zipfile
import cv2 as cv

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate

import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset


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
            # Get images
            image = self.image[idx]
            target = self.target[idx]
            # Random crop
            x = np.random.randint(0, 512-388)
            y = np.random.randint(0, 512-388)
            # Mirror border
            image = mirror_transform(image[x:x+388, y:y+388])
            target = mirror_transform(target[x:x+388, y:y+388])
            # Perform same elastic transformation 
            inp, gt = elastic_transform((image, target), alpha=self.alpha, sigma=self.sigma)
        else:
            # TODO: hardcoded // 512 -> input image size from dataset; 388 -> network's output map
            image_size = self.image[idx].shape[-1]
            target_size = self.target[idx].shape[-1]
            i = idx // 4
            x = (image_size-target_size)*(idx % 2)
            y = (image_size-target_size)*((idx - 4 * i) // 2)
            inp = mirror_transform(self.image[i][x:x+target_size, y:y+target_size])
            gt = mirror_transform(self.target[i][x:x+target_size, y:y+target_size])
        
        gt = gt[92:388+92, 92:388+92]  # crop gt
        _, gt = cv.threshold(gt, 127, 255, cv.THRESH_BINARY)
        gt = gt / 255  # normalize to [0 1]
        inp = (inp - np.min(inp))/np.ptp(inp)  # normalize to [0 1]
        return transforms.ToTensor()(inp.astype('float32')), \
               transforms.ToTensor()(gt).long()



def preprocess_gt(img):
    ''' Preprocessing for the ground truth images (gt):
        This routine highlights object (cell) edges and scales the gt from [0,num_objects] to [0, 255], setting background at 0 and all object at 255.

        Input: 
            - gt: ground truth, generated segmentation mask [0, num_objects]. Torch tensor of shape [H, W].

        Outputs:
            - gt: new ground truth with defined edges between objects (cells). Scaled [0,255]. Torch tensor (same shape as gt).
            - mask_global: edges for objects detected in the gt. Torch tensor (same shape as gt).
    '''
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
    mask_global = np.zeros(img.shape)
    for cls in np.unique(img):
        if cls == 0:  # if not a background
            continue
        mask_cls = np.zeros(img.shape)
        mask_cls[img==cls] = 255  # binary image of cell
        dilated = cv.dilate(mask_cls, kernel, iterations=2)
        mask_global += dilated-mask_cls  # add edge to global mask
        # mask += cv.erode(dilated-thresh,kernel2,iterations=1) 
        # mask += cv.morphologyEx(dilated-thresh, cv.MORPH_OPEN, kernel)
    
    gt = img - mask_global # edges of cells will be background on the new ground truth
    gt[gt<0] = 0  # clipping
    
    return gt, mask_global



def elastic_transform(images, alpha, sigma, random_state=None):
    '''Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       source: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    '''
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
    ''' Fills the difference in size between the original image or patch to a fixed input size (572) of the newtork 
        mirroring the original image outwards.
        
        Inputs:
            - image: original image from the dataset. Numpy array of shape [input_size, input_size].

        Output: 
            - new_image: mirrored image. Numpy array of shape [input_size, input_size].
    '''
    n = len(image)
    input_size = 572
    pad = (input_size - n) // 2
    new_image = np.zeros((input_size, input_size))
    new_image[pad:pad+n,    pad:pad+n] = image
    new_image[:pad,         pad:pad+n] = image[pad:0:-1, 0:n]  # left side
    new_image[:pad,         :pad     ] = new_image[0:pad, pad+pad:pad:-1]  # top left
    new_image[:pad,         pad+n:   ] = new_image[0:pad, n+pad-1:n-1:-1]  # bot left
    new_image[n+pad:,       pad:pad+n] = image[n:n-pad-1:-1, 0:n]  # right side
    new_image[n+pad:,       :pad     ] = new_image[n+pad:, pad+pad:pad:-1]  # top right
    new_image[n+pad:,       n+pad:   ] = new_image[n+pad:, n+pad-1:n-1:-1]  # bot right
    new_image[pad:n+pad,    :pad     ] = image[:, pad:0:-1] # top side
    new_image[pad:n+pad,    n+pad:   ] = image[:, n-1:n-1-pad:-1]  # bot side

    return new_image
    


def mirror_transform_test(image, input_size):
    ''' Fills the difference in size between the original image or patch to the input_size of the newtork mirroring            the original image outwards.
        We shall use this on the test dataset only.
        
        Inputs:
            - image: original image from the dataset. Torch tensor of shape [(1), (1), original_size, original_size].
            - input_size: input size for the network, precomputed by the functions.input_size_compute method.

        Output: 
            - new_image: mirrored image. Torch tensor of shape [1, 1, input_size, input_size].
    '''
    image = image.reshape([image.shape[-1], image.shape[-1]]).numpy()
    n = len(image)
    pad = (input_size - n) // 2
    new_image = np.zeros((input_size, input_size))
    new_image[pad:pad+n,    pad:pad+n] = image
    new_image[:pad,         pad:pad+n] = image[pad:0:-1, 0:n]  # left side
    new_image[:pad,         :pad     ] = new_image[0:pad, pad+pad:pad:-1]  # top left
    new_image[:pad,         pad+n:   ] = new_image[0:pad, n+pad-1:n-1:-1]  # bot left
    new_image[n+pad:,       pad:pad+n] = image[n:n-pad-1:-1, 0:n]  # right side
    new_image[n+pad:,       :pad     ] = new_image[n+pad:, pad+pad:pad:-1]  # top right
    new_image[n+pad:,       n+pad:   ] = new_image[n+pad:, n+pad-1:n-1:-1]  # bot right
    new_image[pad:n+pad,    :pad     ] = image[:, pad:0:-1] # top side
    new_image[pad:n+pad,    n+pad:   ] = image[:, n-1:n-1-pad:-1]  # bot side
    new_image = torch.from_numpy(new_image).reshape([1, 1, input_size, input_size])

    return new_image



def download_data_pkg(cur_dir, dataset_name,  dataset_type):
    folder_name = f"{dataset_name}-{dataset_type}"
    if folder_name not in os.listdir(os.path.join(cur_dir, "data")):
        url = f"http://data.celltrackingchallenge.net/{dataset_type}-datasets/{dataset_name}.zip"
        print(f"Downloading - {folder_name}")
        wget.download(url, cur_dir)
        print(f"Extracting - {folder_name}")
        with zipfile.ZipFile(os.path.join(cur_dir, f"{dataset_name}.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(cur_dir, "data"))
        os.rename(os.path.join(cur_dir, "data", dataset_name), os.path.join(cur_dir, "data", folder_name))
        os.remove(os.path.join(cur_dir, f"{dataset_name}.zip"))
        print(f"Done - {folder_name}")



def download_isbi(cur_dir, dataset_type):
    if dataset_type == "training":
        t = "train"
        folders = ("volume", "labels")
    else:
        t = "test"
        folders = ("volume",)

    folder_name = f"ISBI2012-{dataset_type}"

    if folder_name not in os.listdir(os.path.join(cur_dir, "data")):
        os.mkdir(os.path.join(cur_dir, "data", folder_name))

        for folder in folders:
            print(f"Downloading - {folder_name}_{t}")
            url = f"http://brainiac2.mit.edu/isbi_challenge/sites/default/files/{t}-{folder}.tif"
            wget.download(url, os.path.join(cur_dir, "data", folder_name))
            print(f"Done - {folder_name}")
            if folder == 'volume':
                name = '01'
            else:
                name = '01_GT'

            if name not in os.listdir(os.path.join(cur_dir, "data", folder_name)):
                os.mkdir(os.path.join(cur_dir, "data", folder_name, name))
                if folder == 'labels':
                    os.mkdir(os.path.join(cur_dir, "data", folder_name, name, "SEG"))

                img = Image.open(os.path.join(cur_dir, "data", folder_name, f"{t}-{folder}.tif"))
                i = 0
                while True:
                    try:
                        img.seek(i)
                        if folder == 'volume':
                            img.save(os.path.join(cur_dir, "data", folder_name, name, f't{i:03d}.tif'))
                        else:
                            #  https://stackoverflow.com/questions/29130255/load-tiff-image-as-numpy-array
                            dtype = {'F': np.float32, 'L': np.uint8}[img.mode]

                            # Load the data into a flat numpy array and reshape
                            np_img = np.array(img.getdata(), dtype=dtype)
                            w, h = img.size
                            np_img.shape = (h, w)

                            _, np_img = cv.connectedComponents(np_img)
                            # img.save(os.path.join(cur_dir, "data", folder_name, name, 'SEG', f'man_seg{i:03d}.tif'))
                            cv.imwrite(os.path.join(cur_dir, "data", folder_name, name, 'SEG', f'man_seg{i:03d}.tif'), np_img)
                        i += 1
                    except EOFError:
                        # Not enough frames in img
                        break
    
    

def download_all_data():
    cur_dir = os.path.abspath('')
    if "data" not in os.listdir(cur_dir):
        os.mkdir(os.path.join(cur_dir, "data"))

    download_data_pkg(cur_dir, dataset_name='DIC-C2DH-HeLa', dataset_type='training')
    download_data_pkg(cur_dir, dataset_name='DIC-C2DH-HeLa', dataset_type='challenge')
    download_data_pkg(cur_dir, dataset_name='PhC-C2DH-U373', dataset_type='training')
    download_data_pkg(cur_dir, dataset_name='PhC-C2DH-U373', dataset_type='challenge')

    download_isbi(cur_dir, "training")
    download_isbi(cur_dir, "challenge")


if __name__ == "__main__":
    download_all_data()
