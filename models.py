#### IMPORTS ####
import torch
import torch.nn as nn
from torch.utils import data
from mds189 import Mds189
import numpy as np
from skimage import io, transform
# import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
import pandas as pd


############### SIMPLENET ###############
class SimpLeNet(nn.Module):
    def __init__(self):
        super(SimpLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(int(224 * 448 * 12 / (4 * 4)), 8)
        
    def forward(self, x):
        output = x
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = output.view(-1, np.product(output.size()[1:]))
        output = F.relu(self.fc1(output))
        
        return output


############### ALEXNET ###############
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d()