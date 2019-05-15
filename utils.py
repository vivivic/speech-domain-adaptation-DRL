import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
import scipy.misc
import struct
from hparams import get_hparams

hparams = get_hparams()



class Tanhize(object):
    ''' Normalizing `x` to [-1, 1] '''
    def __init__(self, noise_type):

        max_ = './etc/max_' + noise_type + '.npy'
        min_ = './etc/min_' + noise_type + '.npy'
        
        min_np = np.load(min_)
        max_np = np.load(max_)
        self.xmin = torch.FloatTensor(min_np)
        self.xmax = torch.FloatTensor(max_np)
        self.xscale = self.xmax - self.xmin
    
    def forward_process(self, x):
     
        x = (x - self.xmin) / self.xscale
    
        return torch.clamp(x, 0., 1.) * 2. - 1.

    def backward_process(self, x):
        print(self.xmin.shape)

        return (x* .5 + .5) * self.xscale.cuda() + self.xmin.cuda()

