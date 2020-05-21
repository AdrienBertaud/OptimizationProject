# Pipeline of ResNet model, from image processing to prediction file for submission
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import random
from PIL import Image
from imag_processing import *

from train import *
from pred import *
from generate_submission import *

from PIL import Image
from scipy import ndimage
from dataset import *

import torch.utils.data as utils
import time
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
print(torch.__version__)
print(device)

# Initialize  general parameters
root_dir = "Datasets/training"
num_traset = 100
window_size = 72
patch_size = 16
stride = 16
batch_size = 64
num_workers=0
ratio_tra = 0.8
ratio_test = 1-ratio_tra

# Generate indices randomly for training and validation
indices_tra = random.sample(range(0, num_traset), int(ratio_tra*num_traset))
indices_val = [indices for indices in set(range(0, num_traset)) - set(indices_tra)]

# Load training data
traset_resnet = TrainPatchedDataset(root_dir, num_traset,indices_tra, window_size, 
                                    patch_size, stride, rotate = False, flip = False, shift = False, resnet = True)
# Load test data
valset_resnet = TrainPatchedDataset(root_dir, num_traset, indices_val, window_size, 
                                    patch_size, stride, rotate = False, flip = False, shift = False, resnet = True)

# Load data
num_workers=0
batch_size = 64
train_iter = utils.DataLoader(traset_resnet, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_iter = utils.DataLoader(valset_resnet, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load pretrained model and change the output layer
Resnet34_PT = models.resnet34(pretrained=True)
Resnet34_PT.fc = nn.Linear(512, 2)
print(Resnet34_PT.fc)
net = Resnet34_PT.float()


# Initialize parameters for fine tuning
output_params = list(map(id, Resnet34_PT.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, Resnet34_PT.parameters())
lr, num_epochs = 0.00001, 30
wd = 0.001

optimizer = optim.Adam([{'params': feature_params},
                       {'params': net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay = wd)

# Training ResNet
train_resnet(net, train_iter,val_iter, batch_size, optimizer, device, num_epochs)

# Get prediction image
net.eval()
TEST_SIZE = 50
test_h, test_w = 608, 608
test_data_filename = 'Datasets/test_set_images/'
print("Running prediction on test set")
prediction_test_dir = "predictions_test/"
if not os.path.isdir(prediction_test_dir):
    os.mkdir(prediction_test_dir)
for i in range(1, TEST_SIZE + 1):
    test = "test_%d" % i
    image_filename = test_data_filename + test + '/' + test +  ".png"
    testset = TestPatchedDataset(image_filename, window_size, patch_size, stride, resnet=True)
    test_iter = utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #For test shuffle must be false
    pimg = get_prediction(net,test_iter,window_size,stride,patch_size, test_w, test_h,device)
    pimg8 = img_float_to_uint8(pimg)
    pimg8_L = Image.fromarray(pimg8, 'L')
    pimg8_L.save(prediction_test_dir + "prediction_" + str(i) + ".png")
    
# Get prediction file for submission
generate_submission()