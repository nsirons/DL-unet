import os
import wget
import zipfile
import cv2 as cv

from PIL import Image
import numpy as np


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
