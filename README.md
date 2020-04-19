Reproducibility Project: U-Net
-------------------------------------------------------------------------------------------------------------------------------------------

This repository includes an Ipython notebook written by Group 35 for the reproducibility project in the course of Deep Learning (CS4240), accademic year 2019-2020, TU Delft.

In our notebook we go thought the original paper "U-Net: Convolutional Networks for Biomedical Segmentation" explaining the steps we followed to reproduce the method presented and, particularly, the results obtained with the U-net.
The segmentation results we try to reproduce are the IoU (intersection over union) values got on the datasetes DIC-HeLa and PhC-U373, reported in table 2 and the pixel error obtained on ISBI2012 dataset shown in table 1.

Our goal is a full reproduction of the considered paper: the implementation of the method explained was carried out without using the pre-existing code, made available by the authors.

In the notebook we explain step by step the assumpions we made, our choices and the difference between our implementation and the one described in the paper. 
Finally, we present the obtained results with our reproduction.
For the DIC-HeLa dataset, we obtain an average IoU score of 72.51% that is comparable with the result of 77.56% reported in the paper. Therefore we conclude this result is reproduced.
Using the PhC dataset we get a quite good result of 68.74% IoU. However this is far away for the result of 92.03% the author of the paper obtained. We conclude this result is not reproduced.
For the ISBI2012 dataset, the results we obtained in term of pixel error are far away from the one obtained by the paper: 0.1995 in average against the target result of 0.0582. In this case we conclude this result is not reproduced.

