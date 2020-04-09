import numpy as np
import os

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

import torch
from torchvision.utils import save_image

from functions import evaluation_metrics

from time import time

def testing(unet, test_loader, batch_size, device, output_dir):

    start = time()
    start_eval_test = 0
    idx = 0

    maybe_mkdir_p(os.path.join(output_dir, 'images'))
    maybe_mkdir_p(os.path.join(output_dir, 'preds' ))
    maybe_mkdir_p(os.path.join(output_dir, 'labels'))

    for batch in test_loader:
        
        image, label = batch

        pred = unet(image.to(device))

        pad = int((pred.shape[-1] - label.shape[-1]) / 2)
        pred = pred[:, :, pad:label.shape[-1]+pad, pad:label.shape[-1]+pad].argmax(dim=1)

        save_image(image[0, 0, :, :]     , os.path.join(output_dir, 'images', f'image{idx}.tif'))
        save_image(label[0, 0, :, :].float(), os.path.join(output_dir, 'preds',  f'pred{idx}.tif' ))
        save_image(pred[0, :, :].float() , os.path.join(output_dir, 'labels', f'label{idx}.tif'))

        if start_eval_test == 0: 
            test_eval = evaluation_metrics(pred[0, :, :].detach(), label[0, 0, :, :].detach())
            start_eval_test += 1
        else:
            np.concatenate((test_eval, evaluation_metrics(pred[0, :, :].detach(), label[0, 0, :, :].detach())), axis=1)

    test = np.mean(test_eval, axis=1)

    test_iou = [test[0]]
    test_pe  = [test[1]]
    np.savetxt(os.path.join(output_dir, 'test_iou.out'), test_iou)
    np.savetxt(os.path.join(output_dir, 'test_pe.out' ), test_pe )

    print('Mean IoU testing:', "{:.6f}".format(test[0])          )
    print('Mean PE testing :', "{:.6f}".format(test[1])          )
    print('Testing took    :', "{:.6f}".format(time()-start), 's')
    print('                                                     ')

    print('Testing is finished')