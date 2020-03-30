import os
import wget
import zipfile
import cv2 as cv

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate


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
    
    # Generate field
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    
    # bicubic interpolation
    return (map_coordinates(image, indices, order=3).reshape(shape) for image in images)


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
