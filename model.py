import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
# import ipdb

import numpy as np


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Inatance Normalization"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
       
        b, c = x.size(0), x.size(1)

        
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'




class ResidualBlockAdaIn(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockAdaIn, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=(1,1), padding=1, bias=False),
            AdaptiveInstanceNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=(1,1), padding=1, bias=False),
            AdaptiveInstanceNorm2d(dim_out)
            )

    def forward(self, x):
        return x + self.main(x)



class Discriminator(nn.Module):
    def __init__(
            self,
            ):

        super(Discriminator, self).__init__()

        self.pre_conv = nn.Sequential(
            nn.Conv2d(1, 8, 6, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 6, 2, 2, bias=False),
            nn.InstanceNorm2d(16, affine = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, (1,6), (1,2), (0,2), bias=False),
            nn.InstanceNorm2d(32, affine = True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, (1,3), (1,2), (0,1), bias=False),
            nn.InstanceNorm2d(64, affine = True),
            nn.LeakyReLU(0.2, inplace=True)


            )

        self.dense = nn.Sequential(
                nn.Linear(64*5*5, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,1),
                nn.Sigmoid()
                )

    def forward(self, input):
    
        in_ = self.pre_conv(input.unsqueeze(1))

        in_ = in_.view(-1, 5*5*64)
        out = self.dense(in_)

        # return to origin index 
        return out


class Generator(nn.Module):
    def __init__(
            self,
            extra_layers=False
            ):  
        super(Generator, self).__init__()

        self.pre_fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            ) 


        self.resnet = nn.Sequential(
            ResidualBlockAdaIn(128,128),
            ResidualBlockAdaIn(128,128),
            ResidualBlockAdaIn(128,128),
            ResidualBlockAdaIn(128,128),
            ResidualBlockAdaIn(128,128),
            ResidualBlockAdaIn(128,128),
            )

        self.Tconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (1,3), (1,2), (0,1), (0,1),bias=False),
            nn.InstanceNorm2d(64, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.Tconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (1,3), (1,2), (0,1), (0,1),bias=False),
            nn.InstanceNorm2d(32, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.Tconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (1,6), (1,2), (0,2), bias=False),
            nn.InstanceNorm2d(16, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.Tconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (1,6), (1,2), (0,2), bias=False),
            nn.InstanceNorm2d(8, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.Tconv0 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 1, 1, 0, bias=False),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.Tanh()
            )

    def assign_parameters(self, adain, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":

                mean = adain[:,:128]
                std = adain[:,128:]

                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
        
    def forward(self, in_c, in_s):

        # in_c : [B, 128, T, 10]
        # in_s : [B, 128]

        in_s = in_s
        in_s = self.pre_fc(in_s)

        self.assign_parameters( in_s, self.resnet)

        # in_c : [B, 256, T, 10]

        re3 = self.resnet(in_c)
        # in_c_cnn : [B, 256, T, 10]
        de4 = self.Tconv4(re3)
        de3 = self.Tconv3(de4)
        de2 = self.Tconv2(de3)
        de1 = self.Tconv1(de2)
        out = self.Tconv0(de1)

        return out.squeeze(1)



class ContentEncoder(nn.Module):
    def __init__(
            self,
            extra_layers=False
            ):  
        super(ContentEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1,6), (1,2), (0,2), bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32 , (1,6), (1,2), (0,2), bias=False),
            nn.InstanceNorm2d(32, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64 , (1,3), (1,2), (0,1), bias=False),
            nn.InstanceNorm2d(64, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128 , (1,3), (1,2), (0,1), bias=False),
            nn.InstanceNorm2d(128, affine = True),
            nn.LeakyReLU(0.2, inplace=True)
            )
       


    def forward(self, input):
        in_cnn = input.unsqueeze(1)
        en1 = self.conv1(in_cnn)
        en2 = self.conv2(en1)
        en3 = self.conv3(en2)
        en4 = self.conv4(en3)
  
        return en4


class StyleEncoder(nn.Module):
    def __init__(
            self,
            extra_layers=False
            ):  
        super(StyleEncoder, self).__init__()

        self.conv = nn.Sequential(

                nn.Conv2d(1, 8, (1,6), (1,2), (0,2), bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(8, 16 , (1,6), (1,2), (0,2), bias=False),
                nn.InstanceNorm2d(16, affine = True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(16, 32 , (1,6), (1,2), (0,2), bias=False),
                nn.InstanceNorm2d(32, affine = True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(32, 64 , (1,6), (1,2), (0,2), bias=False),
                nn.InstanceNorm2d(64, affine = True),
                nn.LeakyReLU(0.2, inplace=True),

                nn.AdaptiveMaxPool2d((2,5))
                )

        self.dense = nn.Sequential(
                nn.Linear(128*5, 128),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Linear(32, 8),
                )

    def forward(self, input):

        in_cnn = input.unsqueeze(1)
    
        en1 = self.conv(in_cnn)
        in_dense = en1.view(-1, 64*2*5)

        out = self.dense(in_dense)

        return out

