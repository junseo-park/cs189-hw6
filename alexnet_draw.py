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
from models import *
from torchviz import make_dot

####################### SYS ARGS #######################
import sys
# {key|random} {filestem}
args = sys.argv
print(args)
data_type, filestem = args[1:]
is_key_frame = {'key':True, 'random':False}[data_type]
model_save_name = 'models/' + filestem + '_' + data_type + '.ckpt'

############## Helper functions for loading images. ##############
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


####################### CUDA for PyTorch #######################
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device:', device)
#cudnn.benchmark = True

####################### PARAMS #######################
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 4}
num_epochs = 1
learning_rate = 1e-5
weight_decay = 1e-4

####################### MODEL #######################
model = AlexNet().to(device)
criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

####################### DATASETS #######################
if is_key_frame:
    label_file_train =  'dataloader_files/keyframe_data_train.txt'
    label_file_val  =  'dataloader_files/keyframe_data_val.txt'
else:
    label_file_train = 'dataloader_files/videoframe_data_train.txt'
    label_file_val = 'dataloader_files/videoframe_data_val.txt'
    label_file_test = 'dataloader_files/videoframe_data_test.txt'
    
    
####################### NORMALIZE + DATA GEN #######################
if is_key_frame:
    mean = np.array([134.010302198, 118.599587912, 102.038804945]) / 255
    std = np.array([23.5033438916, 23.8827343458, 24.5498666589]) / 255
else:
    mean = np.array([133.714058398, 118.396875912, 102.262895484]) / 255
    std = np.array([23.2021839891, 23.7064439547, 24.3690056102]) / 255


# Generators
train_dataset = Mds189(label_file_train,loader=default_loader,transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
train_loader = data.DataLoader(train_dataset, **params)

val_dataset = Mds189(label_file_val,loader=default_loader,transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
val_loader = data.DataLoader(val_dataset, **params)


####################### TRAIN/VAL #######################
X = torch.randn_like(torch.unsqueeze(train_dataset[0][0], 0)).to(device)
y = model(X)

g = make_dot(y.mean(), params=dict(model.named_parameters()))
g.format = 'png'
g.render()