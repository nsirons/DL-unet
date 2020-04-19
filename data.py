import os
import wget
import zipfile
import cv2 as cv
import shutil
import requests


from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.stats import norm

import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset

from functions import input_size_compute


class ImageDataset(Dataset):
    def __init__(self, root_dir, alpha=3, sigma=10, crop=388, ISBI2012=False):
        self.root_dir = root_dir
        self.alpha    = alpha
        self.sigma    = sigma
        self.image    = []
        self.target   = []
        self.ISBI2012 = ISBI2012
        self.target_weighted_crop_distribution = []
        
        self.crop = crop
        self.pairs = None
        self.skip = 10
        
        if ISBI2012 is True: 
            n = 1
        else:
            n = len(os.listdir(root_dir)) // 3

        for i in range(1, n+1):
            image_dir = os.path.join(root_dir, f"0{i}")

            if ISBI2012 is True: 
                target_dir = os.path.join(os.path.join(root_dir, f"0{i}_GT", "SEG"))
            else:
                target_dir    = os.path.join(os.path.join(root_dir, f"0{i}_ST", "SEG"))
                target_GT_dir = os.path.join(os.path.join(root_dir, f"0{i}_GT", "SEG"))

                # Remove files from ST (target_dir) present in GT (target_GT_dir) for training
                for image in os.listdir(target_GT_dir):
                    try:
                        os.remove(os.path.join(target_dir, image))
                    except:
                        pass

            image_names = [filename.replace('man_seg', 't') for filename in os.listdir(target_dir)]
            self.image.extend(cv.imread(os.path.join(image_dir, image_name),-1) for image_name in image_names)

            for filename in os.listdir(target_dir):
                img = cv.imread(os.path.join(target_dir, filename), -1)
                gt, _ = preprocess_gt(img)
                _, gt_bin = cv.threshold(gt, 0, 255, cv.THRESH_BINARY)
                self.target.append(gt_bin)
                
                if self.pairs is None:
                    self.pairs = [(ii,jj) for ii in range(0,gt_bin.shape[0] - self.crop,self.skip) for jj in range(0, gt_bin.shape[1] - self.crop,self.skip)]
                
                p = []
                for ii in range(0, gt_bin.shape[0] - self.crop, self.skip):
                    for jj in range(0,gt_bin.shape[1] - self.crop, self.skip):
                        x = np.mean(gt_bin[ii:ii + crop,jj:jj+crop]) / 255
                        if x < 0.1 or x > 0.9:
                            prob = 0
                        else:
                            prob = 10*norm.pdf(x, loc=0.5, scale=0.05)
                        p.append(prob)
                if np.sum(p) == 0:
                    self.target_weighted_crop_distribution.append(np.ones((len(p),)) / len(p))
                else:
                    self.target_weighted_crop_distribution.append(p / np.sum(p))
                        

            # Add files from GT back to ST
            if ISBI2012 is False:
                for image in os.listdir(target_GT_dir):
                    shutil.copyfile(os.path.join(target_GT_dir, image), os.path.join(target_dir, image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # Get images and targets
        image  = np.asarray(self.image[idx])
        target = np.asarray(self.target[idx])
        
        crop_id = np.random.choice(range(len(self.pairs)), 1, p=self.target_weighted_crop_distribution[idx])[0]
        x,y = self.pairs[crop_id]
        x += np.random.randint(-self.skip/2, (self.skip/2) + 1)
        y += np.random.randint(-self.skip/2, (self.skip/2) + 1)
        x = min(max(0, x),target.shape[0] - self.crop)  # not go over image dim
        y = min(max(0, y),target.shape[1] - self.crop)

        image  = image[x:x+self.crop, y:y+self.crop]
        target = target[x:x+self.crop, y:y+self.crop]
        
        original_size = image.shape[-1]
        # Mirror border - Generates image such that network's output size is >= label size
        _, input_size, _ = input_size_compute(image)
        image_pad = np.pad(image, pad_width=input_size, mode='reflect')
        target_pad = np.pad(target, pad_width=input_size, mode='reflect')

        # Random rotation
        rot_deg = np.random.choice(np.arange(0,360,30))
        image_rot = rotate(image_pad, rot_deg)
        target_rot = rotate(target_pad, rot_deg)
        h,w = image_rot.shape
        l = w//2 - input_size//2
        r = w//2 + input_size//2
        t = h//2 - input_size//2
        b = h//2 + input_size//2
        image = image_rot[t:b, l:r]
        target = target_rot[t:b, l:r]
        
        # Perform same elastic transformation 
        inp, gt = elastic_transform((image, target), alpha=self.alpha, sigma=self.sigma)

        pad = int((input_size - original_size) / 2)
        
        gt = gt[pad:original_size+pad, pad:original_size+pad]  # crop gt
        _, gt = cv.threshold(gt, 127, 255, cv.THRESH_BINARY)
        gt = gt / 255  # normalize to [0 1]
        inp = (inp - np.min(inp))/np.ptp(inp)  # normalize to [0 1]
        
        return transforms.ToTensor()(inp.astype('float32')), \
               transforms.ToTensor()(gt).long()



class ImageDataset_test(Dataset):
    def __init__(self, root_dir, ISBI2012=False):
        self.root_dir = root_dir
        self.image    = []
        self.target   = []
        self.ISBI2012 = ISBI2012

        if ISBI2012 is True:
            n = len(os.listdir(root_dir)) - 2 // 2
        else:
            n = len(os.listdir(root_dir)) // 3

        for i in range(1, n+1):
            image_dir = os.path.join(root_dir, f"0{i}")
            target_dir = os.path.join(os.path.join(root_dir, f"0{i}_GT", "SEG"))

            image_names = [filename.replace('man_seg', 't') for filename in os.listdir(target_dir)]
            self.image.extend(cv.imread(os.path.join(image_dir, image_name),-1) for image_name in image_names)

            for filename in os.listdir(target_dir):
                img = cv.imread(os.path.join(target_dir, filename), -1)
                gt, _ = preprocess_gt(img)
                _, gt_bin = cv.threshold(gt, 0, 255, cv.THRESH_BINARY)
                self.target.append(gt_bin)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # Get images and targets
        image = np.asarray(self.image[idx])
        gt    = np.asarray(self.target[idx])

        if image.shape[0] != image.shape[1]: # If images are not square, make them square
            crop = int(abs(image.shape[0] - image.shape[1]) / 2)
            if image.shape[0] > image.shape[1]: # H > W, we should pad H (axis 0)
                image = image[crop:image.shape[1]+crop, :]
                gt    = gt[crop:image.shape[1]+crop, :]
            elif image.shape[0] < image.shape[1]: # H < W, we should pad W (axis 1)
                image = image[:, crop:image.shape[0]+crop]
                gt    = gt[:, crop:image.shape[0]+crop]

        # Apply mirror transform to input image only
        inp = mirror_transform(image)
        
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
    ''' Fills the difference in size between the original image or patch to the input_size of the newtork mirroring            the original image outwards.
        We shall use this on the test dataset only.
        
        Inputs:
            - image: original image from the dataset. Numpy ndarray of shape [original_size, original_size].

        Output: 
            - new_image: mirrored image. Numpy ndarray of shape [input_size, input_size].
    '''
    _, input_size, _ = input_size_compute(image)

    n = image.shape[-1]

    image = image.reshape([n, n])

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
    


def mirror_transform_tensor(image):
    ''' Fills the difference in size between the original image or patch to the input_size of the newtork mirroring            the original image outwards.
        We shall use this on the test dataset only.
        
        Inputs:
            - image: original image from the dataset. Torch tensor of shape [(1), (1), original_size, original_size].

        Output: 
            - new_image: mirrored image. Torch tensor of shape [1, 1, input_size, input_size].
    '''
    _, input_size, _ = input_size_compute(image)

    image = image.reshape([image.shape[-1], image.shape[-1]]).numpy()

    n = image.shape[-1]

    image = image.reshape([n, n])

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


def download_file_from_google_drive(id, destination):
    """
    source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_all_models():
    cur_dir = os.path.abspath('')
    if "models" not in os.listdir(cur_dir):
        os.mkdir(os.path.join(cur_dir, "models"))
    
    models = (
        ('ISBI2012','1tivQbiNkaQLlEN5ck5JYHyH1hczH_Kgq'),
        ('DIC-C2DH-HeLa','1Fn5_wSYEFX50orh_qYWDc2BkWjCByRmC'),
        ('PhC-C2DH-U373','1SWuBGSgQJvR2yBZpR4CSEoOQj2_pGfVI'),
    )
    for name, file_id in models:
        if name not in os.listdir(os.path.join(cur_dir, 'models')):
            print(f"Downloading model - {name}")
            destination = os.path.join(cur_dir, 'models', name + '.zip')
            download_file_from_google_drive(file_id, destination)
            print(f"Extracting mode - {name}")
            with zipfile.ZipFile(destination, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(cur_dir, "models"))
            os.remove(destination)
            print(f"Done - {name}")

if __name__ == "__main__":
    download_all_data()
    download_all_models()
