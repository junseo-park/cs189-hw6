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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib

def plot_confusion_matrix(y_test, y_pred, label_map, filestem, data_type):
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cbar=False, cmap=matplotlib.cm.get_cmap('gist_yarg'))
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    plt.xticks(np.arange(cm.shape[1]) + 0.5, label_map)
    plt.yticks(np.arange(cm.shape[0]) + 0.5, label_map, rotation='horizontal')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.savefig('img/' + filestem + '_' + data_type + '_confmat.png')


############### SIMPLENET1 ###############
class SimpLeNet1(nn.Module):
    def __init__(self):
        super(SimpLeNet1, self).__init__()
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
    
############### SIMPLENET2 ############### (simplenet1 + fc2)
class SimpLeNet2(nn.Module):
    def __init__(self):
        super(SimpLeNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(int(224 * 448 * 12 / (4 * 4)), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 8)
        
    def forward(self, x):
        output = x
        output = self.pool(F.relu(self.conv1(output)))
        output = self.pool(F.relu(self.conv2(output)))
        output = output.view(-1, np.product(output.size()[1:]))
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        
        return output

############### SIMPLENET3 ############### (example script)
class SimpLeNet3(nn.Module):
    def __init__(self):
        super(SimpLeNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(int(224 * 448 * 48 / (4 * 4)), 8)
        
    def forward(self, x):
        output = x
        output = F.relu(self.conv1(output))
        output = self.pool(F.relu(self.conv2(output)))
        output = F.relu(self.conv3(output))
        output = self.pool(F.relu(self.conv4(output)))
        output = output.view(-1, np.product(output.size()[1:]))
        output = F.relu(self.fc1(output))
        
        return output

############### ALEXNET ###############
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 8)
        )
        
    def forward(self, x):
        output = x
        output = self.conv(output)
        output = self.avgpool(output)
        output = output.view(-1, 256 * 6 * 6)
        output = self.fc(output)
        
        return output