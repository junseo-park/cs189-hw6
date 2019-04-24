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
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}
num_epochs = 15
learning_rate = 1e-5
weight_decay = 0.5

####################### MODEL #######################
model = SimpLeNet1().to(device)
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
start = time.time()
print('Beginning training..')
total_step = len(train_loader)
train_loss_list = []
val_loss_list = []

for epoch in range(num_epochs):
    #### TRAINING ####
    print('epoch {}'.format(epoch))
    for i, (local_batch,local_labels) in enumerate(train_loader):
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        # Forward pass
        outputs = model.forward(local_ims)
        train_loss = criterion(outputs, local_labels)
        train_loss_list.append(train_loss)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (i+1) % 4 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, train_loss.item()))
    
    #### VALDATION ####
    running_val_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_list = []
        groundtruth_list = []
        for i, (local_batch, local_labels) in enumerate(val_loader):
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            
            # Forward pass
            outputs = model.forward(local_ims)
            val_loss = criterion(outputs, local_labels)
            running_val_loss += val_loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            predicted_list.extend(predicted)
            groundtruth_list.extend(local_labels)
            correct += (predicted == local_labels).sum().item()
        print('------------------------------------------')
        print('VALIDATION LOSS, Epoch [{}/{}]: {:.4f}'
             .format(epoch + 1, num_epochs, running_val_loss / i))
        val_loss_list.append(running_val_loss / i)
        
        
        print('VALIDATION ACCURACY, EPOCH [{}/{}]: {:.4f}%'.format(epoch + 1, num_epochs, 100 * correct / total))
        print('------------------------------------------')
        
        

end = time.time()
print('Training time: {}'.format(end - start))


# Save the model checkpoint
torch.save(model.state_dict(), model_save_name)

plt.figure(figsize=(10, 7))
plt.plot(np.arange(1, len(train_loss_list) + 1), train_loss_list, label='train')
plt.plot(np.arange(1, len(train_loss_list) + 1, total_step), val_loss_list, label='val')
plt.legend()
plt.title(filestem + ' Loss Curve')
plt.xlabel(f'Step ({total_step} steps/epoch)')
plt.ylabel('Cross-Entropy Loss')
plt.savefig('img/' + filestem + '_' + data_type + '_loss' '.png')

# Look at some things about the model results..
# convert the predicted_list and groundtruth_list Tensors to lists
pl = [p.cpu().numpy().tolist() for p in predicted_list]
gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

# TODO: use pl and gt to produce your confusion matrices
label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
plot_confusion_matrix(gt, pl, label_map, filestem, data_type)


# view the per-movement accuracy
label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
for id in range(len(label_map)):
    print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))