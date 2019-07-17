import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image
from prune import PruningModule, MaskedConv2d


class SRCNN(PruningModule):
    def __init__(self, mask=True):
        super(SRCNN,self).__init__()
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = conv2d(3,64,kernel_size=(9,9),padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = conv2d(64,32,kernel_size=(5,5),padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = conv2d(32,3,kernel_size=(5,5),padding=2)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out