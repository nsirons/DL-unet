import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__() # (what does this do?)
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

        # self.upsamp4 = nn.UpsamplingBilinear2d(size=(28,28), scale_factor=2) # Do we have to input each channel separately? 2d or 3d?
        # self.upconv4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2)

        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.conv41e = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv42e = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        # self.upsamp3 = nn.UpsamplingBilinear2d(size=(52,52), scale_factor=2) 
        # self.upconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2)

        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.conv31e = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv32e = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        # self.upsamp2 = nn.UpsamplingBilinear2d(size=(100,100), scale_factor=2) 
        # self.upconv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2)

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.conv21e = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv22e = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        # self.upsamp1 = nn.UpsamplingBilinear2d(size=(196,196), scale_factor=2) 
        # self.upconv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2)

        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv11e = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv12e = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.finalconv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def crop_and_concat(self, A, B):
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
        # t = self.upsamp4(t) # Dealing with dimensions
        t = self.upconv4(t)
        t = self.crop_and_concat(t4, t)

        # Fourth level expansive convolutions

        t = F.relu(self.conv41e(t))
        t = F.relu(self.conv42e(t))

        # Upconvolution 3
        # t = self.upsamp3(t) # Dealing with dimensions
        t = self.upconv3(t)
        t = self.crop_and_concat(t3, t)

        # Third level expansive convolutions

        t = F.relu(self.conv31e(t))
        t = F.relu(self.conv32e(t))

        # Upconvolution 2
        # t = self.upsamp2(t) # Dealing with dimensions
        t = self.upconv2(t)
        t = self.crop_and_concat(t2, t)

        # Second level expansive convolutions

        t = F.relu(self.conv21e(t))
        t = F.relu(self.conv22e(t))

        # Upconvolution 1
        # t = self.upsamp1(t) # Dealing with dimensions
        t = self.upconv1(t)
        t = self.crop_and_concat(t1, t)

        # First level expansive convolutions

        t = F.relu(self.conv11e(t))
        t = F.relu(self.conv12e(t))

        t = self.finalconv(t)

        return t