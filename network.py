import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class Unet(nn.Module):

    """ Unet 2D implementation (Ronneberger et al. 2015)

    This network reproduces the Unet architecture described in Ronneberger et al. 2015.
    The structure of this network is made up of 23 convolutional layers including pooling 
    operation and ReLU activation functions, as well as transpose convolutions. We can 
    identify a contractive path and an expansive path with skip connections between both.

    Possible improvements: BN, padding to conv layers
    """

    def __init__(self):
        super(Unet, self).__init__() 

        self.conv11c = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv12c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv21c = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv22c = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv31c = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv32c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.conv41c = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv42c = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.conv51c = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv52c = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.conv41e = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv42e = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.conv31e = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv32e = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv21e = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv22e = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv11e = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv12e = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.finalconv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

        """ Initialization of the weights

        We follow here the criteria provided by Ronneberger et al. (2015). Accordingly,
        the weights should be initialized following a gaussian distribution of mean 0 
        and standard deviation std = (2 / N) ** 0.5 with N being the number of incoming 
        nodes of one neuron.

        Example:
        For a 3x3 convolution and 64 feature channels in the previous layer, N = 3 ** 2 * 64
        """
        self.conv11c.weight = nn.Parameter(torch.empty_like(self.conv11c.weight).normal_(mean = 0, std = (2 ** 0.5)))
        self.conv12c.weight = nn.Parameter(torch.empty_like(self.conv12c.weight).normal_(mean = 0, std = (2 / (self.conv12c.in_channels * float(self.conv11c.kernel_size[0]) ** 2) ** 0.5)))

        self.conv21c.weight = nn.Parameter(torch.empty_like(self.conv21c.weight).normal_(mean = 0, std = (2 / (self.conv21c.in_channels * float(self.conv12c.kernel_size[0]) ** 2) ** 0.5)))
        self.conv22c.weight = nn.Parameter(torch.empty_like(self.conv22c.weight).normal_(mean = 0, std = (2 / (self.conv22c.in_channels * float(self.conv21c.kernel_size[0]) ** 2) ** 0.5)))

        self.conv31c.weight = nn.Parameter(torch.empty_like(self.conv31c.weight).normal_(mean = 0, std = (2 / (self.conv31c.in_channels * float(self.conv22c.kernel_size[0]) ** 2) ** 0.5)))
        self.conv32c.weight = nn.Parameter(torch.empty_like(self.conv32c.weight).normal_(mean = 0, std = (2 / (self.conv32c.in_channels * float(self.conv31c.kernel_size[0]) ** 2) ** 0.5)))

        self.conv41c.weight = nn.Parameter(torch.empty_like(self.conv41c.weight).normal_(mean = 0, std = (2 / (self.conv41c.in_channels * float(self.conv32c.kernel_size[0]) ** 2) ** 0.5)))
        self.conv42c.weight = nn.Parameter(torch.empty_like(self.conv42c.weight).normal_(mean = 0, std = (2 / (self.conv42c.in_channels * float(self.conv41c.kernel_size[0]) ** 2) ** 0.5)))

        self.conv51c.weight = nn.Parameter(torch.empty_like(self.conv51c.weight).normal_(mean = 0, std = (2 / (self.conv51c.in_channels * float(self.conv42c.kernel_size[0]) ** 2) ** 0.5)))
        self.conv52c.weight = nn.Parameter(torch.empty_like(self.conv52c.weight).normal_(mean = 0, std = (2 / (self.conv52c.in_channels * float(self.conv51c.kernel_size[0]) ** 2) ** 0.5)))

        self.upconv4.weight = nn.Parameter(torch.empty_like(self.upconv4.weight).normal_(mean = 0, std = (2 / (self.upconv4.in_channels * float(self.conv52c.kernel_size[0]) ** 2) ** 0.5)))

        self.conv41e.weight = nn.Parameter(torch.empty_like(self.conv41e.weight).normal_(mean = 0, std = (2 / (self.conv42c.out_channels * float(self.conv42c.kernel_size[0]) ** 2 + self.upconv4.out_channels * float(self.upconv4.kernel_size[0]) ** 2) ** 0.5)))
        self.conv42e.weight = nn.Parameter(torch.empty_like(self.conv42e.weight).normal_(mean = 0, std = (2 / (self.conv42e.in_channels * float(self.conv41e.kernel_size[0]) ** 2) ** 0.5)))

        self.upconv3.weight = nn.Parameter(torch.empty_like(self.upconv3.weight).normal_(mean = 0, std = (2 / (self.upconv3.in_channels * float(self.conv42e.kernel_size[0]) ** 2) ** 0.5)))

        self.conv31e.weight = nn.Parameter(torch.empty_like(self.conv31e.weight).normal_(mean = 0, std = (2 / (self.conv32c.out_channels * float(self.conv32c.kernel_size[0]) ** 2 + self.upconv3.out_channels * float(self.upconv3.kernel_size[0]) ** 2) ** 0.5)))
        self.conv32e.weight = nn.Parameter(torch.empty_like(self.conv32e.weight).normal_(mean = 0, std = (2 / (self.conv32e.in_channels * float(self.conv31e.kernel_size[0]) ** 2) ** 0.5)))

        self.upconv2.weight = nn.Parameter(torch.empty_like(self.upconv2.weight).normal_(mean = 0, std = (2 / (self.upconv2.in_channels * float(self.conv32e.kernel_size[0]) ** 2) ** 0.5)))

        self.conv21e.weight = nn.Parameter(torch.empty_like(self.conv21e.weight).normal_(mean = 0, std = (2 / (self.conv22c.out_channels * float(self.conv22c.kernel_size[0]) ** 2 + self.upconv2.out_channels * float(self.upconv2.kernel_size[0]) ** 2) ** 0.5)))
        self.conv22e.weight = nn.Parameter(torch.empty_like(self.conv22e.weight).normal_(mean = 0, std = (2 / (self.conv22e.in_channels * float(self.conv21e.kernel_size[0]) ** 2) ** 0.5)))

        self.upconv1.weight = nn.Parameter(torch.empty_like(self.upconv1.weight).normal_(mean = 0, std = (2 / (self.upconv1.in_channels * float(self.conv22e.kernel_size[0]) ** 2) ** 0.5)))

        self.conv11e.weight = nn.Parameter(torch.empty_like(self.conv11e.weight).normal_(mean = 0, std = (2 / (self.conv12c.out_channels * float(self.conv12c.kernel_size[0]) ** 2 + self.upconv1.out_channels * float(self.upconv1.kernel_size[0]) ** 2) ** 0.5)))
        self.conv12e.weight = nn.Parameter(torch.empty_like(self.conv12e.weight).normal_(mean = 0, std = (2 / (self.conv12e.in_channels * float(self.conv11e.kernel_size[0]) ** 2) ** 0.5)))

        self.finalconv.weight = nn.Parameter(torch.empty_like(self.finalconv.weight).normal_(mean = 0, std = (2 / (self.finalconv.in_channels * float(self.conv12e.kernel_size[0]) ** 2) ** 0.5)))
        

    def crop_and_concat(self, A, B):
        
        """ Crop and concatenation (for skip connections)
            
            We use this simple function to crop the tensor (H and W) from the contractive 
            path to the size of that from the expansive path and concatenate both. We need 
            this method for the skip connections found in the Unet architecture.

            Inputs:
                - A: tensor from the contractive path.
                - B: tensor from the expansive path.

            Output:
                - Concatenated (axis=1) tensor 
        """

        crop_factor=(A.size()[2] - B.size()[2]) * 0.5
        c=int(crop_factor)
        A = F.pad(A, (-c, -c, -c, -c))
        return torch.cat((A,B), 1)

    def forward(self,t):
        # First level contractive convolutions
        t = F.relu(self.conv11c(t))
        t = F.relu(self.conv12c(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t1 = t

        # Second level contractive convolutions
        t = F.relu(self.conv21c(t))
        t = F.relu(self.conv22c(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t2 = t

        # Third level contractive convolutions
        t = F.relu(self.conv31c(t))
        t = F.relu(self.conv32c(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t3 = t

        # Fourth level contractive convolutions
        t = F.relu(self.conv41c(t))
        t = F.relu(self.conv42c(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t4 = t

        # Fifth level contractive convolutions
        t = F.relu(self.conv51c(t))
        t = F.relu(self.conv52c(t))

        # Upconvolution 4
        t = self.upconv4(t)
        t = self.crop_and_concat(t4, t)

        # Fourth level expansive convolutions
        t = F.relu(self.conv41e(t))
        t = F.relu(self.conv42e(t))

        # Upconvolution 3
        t = self.upconv3(t)
        t = self.crop_and_concat(t3, t)

        # Third level expansive convolutions
        t = F.relu(self.conv31e(t))
        t = F.relu(self.conv32e(t))

        # Upconvolution 2
        t = self.upconv2(t)
        t = self.crop_and_concat(t2, t)

        # Second level expansive convolutions
        t = F.relu(self.conv21e(t))
        t = F.relu(self.conv22e(t))

        # Upconvolution 1
        t = self.upconv1(t)
        t = self.crop_and_concat(t1, t)

        # First level expansive convolutions
        t = F.relu(self.conv11e(t))
        t = F.relu(self.conv12e(t))

        t = self.finalconv(t)

        return t
