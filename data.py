import os
import wget
import zipfile
import cv2


import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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


def download_all_data():
    cur_dir = os.path.abspath('')
    if "data" not in os.listdir(cur_dir):
        os.mkdir(os.path.join(cur_dir, "data"))

    download_data_pkg(cur_dir, dataset_name='DIC-C2DH-HeLa', dataset_type='training')
    download_data_pkg(cur_dir, dataset_name='DIC-C2DH-HeLa', dataset_type='challenge')
    download_data_pkg(cur_dir, dataset_name='PhC-C2DH-U373', dataset_type='training')
    download_data_pkg(cur_dir, dataset_name='PhC-C2DH-U373', dataset_type='challenge')


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       source: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image = []
        self.target = []

        n = len(os.listdir(root_dir)) // 3
        for i in range(1, n+1):
            image_folder = os.path.join(root_dir, f"0{i}")
            target_folder = os.path.join(os.path.join(root_dir, f"0{i}_GT", "SEG"))
            image_names = [filename.replace('man_seg', 't') for filename in os.listdir(target_folder)]
            self.image.extend(cv2.imread(os.path.join(image_folder, image_name), -1) for image_name in image_names)
            for filename in os.listdir(target_folder):
                img = cv2.imread(os.path.join(target_folder, filename), -1)
                img[img > 0] = 1.
                self.target.append(img)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        return transforms.ToTensor()(elastic_transform(self.image[idx], alpha=3, sigma=10)), \
               transforms.ToTensor()(elastic_transform(self.target[idx], alpha=3, sigma=10))


# import cv2
# import numpy as np
# cur_dir = os.path.abspath('')
# print(os.path.join(cur_dir, "data", "DIC-C2DH-HeLa-training", "01_GT", "SEG", "man_seg067.tif"))
# img = cv2.imread(os.path.join(cur_dir, "data", "DIC-C2DH-HeLa-training", "01_GT", "SEG", "man_seg005.tif"), -1)
# img_scaled = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
# print(np.max(img_scaled))
# cv2.imshow("hello", img_scaled)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == '__main__':
    import cv2

    import numpy as np
    cur_dir = os.path.abspath('')
    print(os.path.join(cur_dir, "data", "DIC-C2DH-HeLa-training", "01_GT", "SEG", "man_seg067.tif"))
    img = cv2.imread(os.path.join(cur_dir, "data", "DIC-C2DH-HeLa-training", "01_GT", "SEG", "man_seg005.tif"), -1)
    # img_scaled = cv2.normalize(img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    img[img > 0] = 65535
    img_scaled = img
    print(np.max(img_scaled), np.min(img_scaled))
    cv2.imshow("hello", img_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
    # run this to download data
    download_all_data()

    cur_dir = os.path.abspath('')
    root_dir = os.path.join(cur_dir, "data", "DIC-C2DH-HeLa-training")
    transformed_dataset = ImageDataset(root_dir)

    # Testing
    img, mask = transformed_dataset[0]
    total = np.hstack((img.numpy().reshape((512, 512)), mask.numpy().reshape(512,512)))
    cv2.imshow("hello", total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
