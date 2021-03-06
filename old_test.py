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

import sys
# {key|random} {filestem}
args = sys.argv
print(args)
data_type, filestem = args[1:]
is_key_frame = {'key':True, 'random':False}[data_type]
model_to_load = filestem + '.ckpt'



#### Helper functions for loading images. ####
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
          'shuffle': False,
          'num_workers': 4}
num_epochs = 50
learning_rate = 1e-5



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
val_dataset = Mds189(label_file_val,loader=default_loader,transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
val_loader = data.DataLoader(val_dataset, **params)

if not is_key_frame:
    test_dataset = Mds189(label_file_test,loader=default_loader,transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)
                                               ]))
    test_loader = data.DataLoader(test_dataset, **params)
    

####################### TEST #######################
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print('Beginning Testing..')
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    groundtruth_list = []
    for (local_batch,local_labels) in test_loader:
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = model.forward(local_ims)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        predicted_list.extend(predicted)
        groundtruth_list.extend(local_labels)
        correct += (predicted == local_labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

if run_type == 'val':
    # Look at some things about the model results..
    # convert the predicted_list and groundtruth_list Tensors to lists
    pl = [p.cpu().numpy().tolist() for p in predicted_list]
    gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

    # TODO: use pl and gt to produce your confusion matrices
    label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
    def plot_confusion_matrix(y_test, y_pred):
        plt.figure(figsize=(10, 7))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cbar=False, cmap=matplotlib.cm.get_cmap('gist_yarg'))
        plt.ylabel('Observed')
        plt.xlabel('Predicted')
        plt.xticks(cm.shape[1], label_map)
        plt.yticks(cm.shape[0], label_map, rotation='horizontal')
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.savefig('img/' + filestem + data_type + '.png')


    # view the per-movement accuracy
    label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
    for id in range(len(label_map)):
        print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))

# TODO: you'll need to run the forward pass on the kaggle competition images, and save those results to a csv file.
if run_type == 'test' and not is_key_frame:
    jpgs = [f'{i:03}.jpg' for i in range(len(pl))]
    df = pd.DataFrame({'Id':jpgs, 'Category':pl})
    df['Category'] = df['Category'].map(dict(zip(np.arange(label_map), label_map)))
    df.to_csv('submission_' + filestem + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False)