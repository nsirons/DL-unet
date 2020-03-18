import cv2 as cv
import numpy as np
import torch

def weighted_map(gt_batch, batch_size):

    """ This method is needed to compute the weight map for each ground truth (gt) segmentation, to compensate the different frequency of pixels for each class. 
        With this method, we highlight the borders between different objects (cells in the case of the HeLa dataset used in Ronneberger et al. 2015) 

        Input:
            - gt: ground truth, generated segmentation mask (0,1). Torch tensor os shape [batch_size, H, W]

        Output: 
            - w: weight map. Torch tensor (same shape as gt)

    """

    w_batch = torch.empty_like(gt_batch)

    for batch_pos in range(batch_size):

        gt = gt_batch[batch_pos, :, :]

        # Hyperparameters given by Ronneberger et al. 2015
        w0 = 10
        sig2 = 25 # 5^2

        # The unique command returns the unique values inside the input tensor, in these case 0 and 1, and these are stored in "uval"
        # "counts" includes the number of times we find each value in "uval" inside of the input tensor
        [uval, counts] = torch.unique(gt, return_counts=True)

        # For the coputation of the w_c tensor:
        w_c = torch.empty(gt.shape)
        for pos in range(len(uval)):
            # We normalize all pixel values according to the class frequencies. We also constrain that the cell class is set to 1
            w_c[gt == uval[pos]] = counts[1].float() / counts[pos].float() 

        # Now we have to highlight borders with the second element from formula (2) from Ronneberger et al. (2015)
        gt_img = gt.view(gt.shape[1], -1)
        gt_img = gt_img.numpy()

        # Identifies separated objects present in gt and gives each one a different value
        n_obj, objects = cv.connectedComponents(gt_img.astype(np.uint8), connectivity=4)

        # In order to compute the distance map for each object, we separate these on different channels
        objects_sep = np.zeros([n_obj-1, gt_img.shape[0], gt_img.shape[1]])
        for ii in range(n_obj-1):
            objects_sep[ii, :, :] = objects==ii+1

        # Computes the distance transform, i.e. distance to the closest zero-value pixel for each identified object
        # Notice that we are subtracting 1 from 
        maps = np.zeros([n_obj-1, gt_img.shape[0], gt_img.shape[1]])
        for ii in range(n_obj-1):
            maps[ii,: , :] = cv.distanceTransform(objects_sep[ii, :, :].astype(np.uint8)-1, cv.DIST_L2, maskSize=0)

        # We sort the distance maps along the channels (objects) and we then keep d1 and d2, defined as the distance the 
        # closest two objects for each background pixel
        maps = np.sort(maps, 0)
        d1 = maps[0, :, :]
        d2 = maps[1, :, :]

        # We now compute the w_d matrix following formula (2) in Ronneberger et al. 2015
        w_d = np.zeros(gt_img.shape)
        w_d = np.multiply(w0 * np.exp(-np.square(d1+d2) / (2 * sig2)), -1 * (gt - 1)) # We only compute background pixels

        # Finally we add w_c and w_d to generate the final weights
        w = w_c + w_d

        # Convert to PyTorch tensor
        w_batch[batch_pos, :, :] = w.clone().detach()

    return w_batch