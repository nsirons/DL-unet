import cv2 as cv
import numpy as np
import torch

# from sklearn.metrics import f1_score

def weighted_map(gt_batch):

    """ This method is needed to compute the weight map for each ground truth (gt) segmentation, to compensate the different frequency of pixels for each class. 
        With this method, we highlight the borders between different objects (cells in the case of the HeLa dataset used in Ronneberger et al. 2015) 

        Input:
            - gt: ground truth, generated segmentation mask (0,1). Torch tensor os shape [batch_size, H, W]

        Output: 
            - w: weight map. Torch tensor (same shape as gt)

    """

    w_batch = torch.empty_like(gt_batch)

    batch_size = gt_batch.shape[0]

    for batch_pos in range(batch_size):

        gt = gt_batch[batch_pos, :, :]

        # Hyperparameters given by Ronneberger et al. 2015
        w0 = 20
        sig2 = 25 # 5^2

        # The unique command returns the unique values inside the input tensor, in these case 0 and 1, and these are stored in "uval"
        # "counts" includes the number of times we find each value in "uval" inside of the input tensor
        [uval, counts] = torch.unique(gt, return_counts=True)

        # For the coputation of the w_c tensor:
        w_c = torch.empty_like(gt)
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
        d2 = 0
        if maps.shape[0] > 1:
            d2 = maps[1, :, :]

        # We now compute the w_d matrix following formula (2) in Ronneberger et al. 2015
        w_d = np.zeros(gt_img.shape)
        w_d = np.multiply(w0 * np.exp(-np.square(d1+d2) / (2 * sig2)), -1 * (gt - 1)) # We only compute background pixels

        # Finally we add w_c and w_d to generate the final weights
        w = w_c + w_d

        # Convert to PyTorch tensor
        w_batch[batch_pos, :, :] = w.clone().detach()

    return w_batch



def class_balance(gt_batch):

    """ This method is needed to compute the weight map for each ground truth (gt) segmentation, to compensate the different frequency of pixels for each class. 

        Input:
            - gt: ground truth, generated segmentation mask (0,1). Torch tensor os shape [batch_size, H, W]

        Output: 
            - w: weight map. Torch tensor (same shape as gt)

    """

    w_batch = torch.empty_like(gt_batch)

    batch_size = gt_batch.shape[0]

    for batch_pos in range(batch_size):

        gt = gt_batch[batch_pos, :, :]

        # The unique command returns the unique values inside the input tensor, in these case 0 and 1, and these are stored in "uval"
        # "counts" includes the number of times we find each value in "uval" inside of the input tensor
        [uval, counts] = torch.unique(gt, return_counts=True)

        # For the coputation of the w_c tensor:
        w_c = torch.empty_like(gt)
        for pos in range(len(uval)):
            # We normalize all pixel values according to the class frequencies. We also constrain that the cell class is set to 1
            w_c[gt == uval[pos]] = counts[1].float() / counts[pos].float() 

        # Convert to PyTorch tensor
        w_batch[batch_pos, :, :] = w_c.clone().detach()

    return w_batch



def input_size_compute(image):
    ''' Computes what the network's input image size should be so that it outputs the smallest image with a size over the
        original image. The difference between original image's size and network's input image should be mirrored.

        Input: 
            - image: original image from the dataset. Expects Tensor of shape [(batch_size), (num_channels), original_size, original_size].

        Outputs:
            - original_size: size (H or W) of the original image from the dataset.
            - input_size: size (H or W) of the network's input image.
            - output_size: size (H or W) of the network's output image.
    '''
    original_size = image.shape[-1]
    lowest_res = 20

    # We compute a viable input size by setting the lowest resolution in the network's processing.
    input_size  = (((lowest_res * 2 + 4) * 2 + 4) * 2 + 4) * 2 + 4
    output_size = ((((lowest_res - 4) * 2 - 4) * 2 - 4) * 2 - 4) * 2 - 4

    while output_size < original_size:
        lowest_res += 2

        input_size  = (((lowest_res * 2 + 4) * 2 + 4) * 2 + 4) * 2 + 4
        output_size = ((((lowest_res - 4) * 2 - 4) * 2 - 4) * 2 - 4) * 2 - 4

    return original_size, input_size, output_size



def evaluation_metrics(pred, label):
    ''' Intersection over Union (IoU) and pixel error (or squared Euclidean distance) between labels and predictions.
        Inputs:
            - preds: prediction tensor. Torch tensor of shape [(batch_size), H, W].
            - labels: labels tensor. Torch tensor of shape [(batch_size), H, W].

        Output:
            - iou: float.
            - pixel_error: float.
    '''

    iou = IoU(pred, label)
    
    pixel_error = Pixel_error(pred, label)

    evaluation_metrics = np.empty([2,1])

    evaluation_metrics[0] = iou
    evaluation_metrics[1] = pixel_error

    return evaluation_metrics



def Pixel_error(pred, label):
    ''' Computes the pixel error of the network's prediction compared to the labels corresponding to the same image 
    (squared Euclidean distance).
        
    Inputs:
        - preds: prediction tensor. Torch tensor of shape [(batch_size), H, W].
        - labels: labels tensor. Torch tensor of shape [(batch_size), H, W]. 

    Outputs:
        - pixel_error: float.
    '''
    pred_np  = pred.cpu().numpy()
    label_np = label.cpu().numpy()

    pixel_error = np.linalg.norm(pred_np - label_np) / pred_np.size

    return pixel_error



def IoU(pred, label):
    ''' Intersection over Union (IoU) between labels and predictions.
        Inputs:
            - pred: prediction tensor. Torch tensor of shape [(batch_size), H, W].
            - label: labels tensor. Torch tensor of shape [(batch_size), H, W].

        Output:
            - iou: float.
    '''
    pred_np = pred.cpu().numpy()
    label_np = label.cpu().numpy()

    # dice = np.sum(pred_np[label_np==1]==1)*2.0 / (np.sum(pred_np==1) + np.sum(label_np==1))

    intersection = np.logical_and(pred_np, label_np)
    union        = np.logical_or(pred_np,  label_np)

    iou = np.sum(intersection) / np.sum(union)

    return iou