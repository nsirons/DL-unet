import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class Unet(nn.Module):
    def __init__(self):
        super(Network, self).__init__() # (what does this do?)
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

        self.upsamp1 = nn.UpsamplingBilinear2d(size=(28,28), scale_factor=2) # Do we have to input each channel separately? 2d or 3d?
        self.upconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=2)

        

    def crop_and_concat(self, A, B):
        crop_factor=(A.size()[2] - B.size()[2]) * 0.5
        A = F.pad(A, (-crop_factor, -crop_factor, -crop_factor, -crop_factor))
        return torch.cat((A,B), 1)

    def forward(self,t):

        # First level contractive convolutions
        t = F.relu(self.conv11c(t))
        t = F.relu(self.conv12c(t))
        t = self.F.max_pool2d(t, kernel_size=2, stride=2)
        t1 = t

        # Second level contractive convolutions
        t = F.relu(self.conv21c(t))
        t = F.relu(self.conv22c(t))
        t = self.F.max_pool2d(t, kernel_size=2, stride=2)
        t2 = t

        # Third level contractive convolutions
        t = F.relu(self.conv31c(t))
        t = F.relu(self.conv32c(t))
        t = self.F.max_pool2d(t, kernel_size=2, stride=2)
        t3 = t

        # Fourth level contractive convolutions
        t = F.relu(self.conv41c(t))
        t = F.relu(self.conv42c(t))
        t = self.F.max_pool2d(t, kernel_size=2, stride=2)
        t4 = t

        # Fifth level contractive convolutions
        t = F.relu(self.conv51c(t))
        t = F.relu(self.conv52c(t))
        t = self.F.max_pool2d(t, kernel_size=2, stride=2)

        # Upconvolution 1
        t = upsamp1

