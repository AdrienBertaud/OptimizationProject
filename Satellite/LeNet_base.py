# Pipeline of baseline model LeNet, from image processing to prediction file for submission
import matplotlib.image as mpimg
import numpy as np
import os,sys
from imag_processing import *
from patch_prepare import *

from train import *
from pred import *
from dataset import *
from generate_submission import *

from PIL import Image
from scipy import ndimage
from imag_processing import *
import random

import torch.utils.data as utils
import time
import torch
from torch import nn, optim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print("torch version : ", torch.__version__)
print("device : ", device)

# Initialize parameters
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
traset = TrainPatchedDataset(root_dir, num_traset,indices_tra, window_size, patch_size,
                                    stride, rotate = False, flip = False, shift = False, resnet = False)

# Load test data
valset = TrainPatchedDataset(root_dir, num_traset, indices_val, window_size, patch_size, 
                                    stride, rotate = False, flip = False, shift = False, resnet = False)

# Define dataloader for training and validation
train_iter = utils.DataLoader(traset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_iter = utils.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*15*15, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
# Initialize parameters for training
net = LeNet().float()
lr, num_epochs = 0.0001, 1
wd = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# Training
train_cnn(net, train_iter, val_iter, batch_size, optimizer, device, num_epochs)



# Get prediction image
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
    testset = TestPatchedDataset(image_filename, window_size, patch_size, stride, resnet=False)
    test_iter = utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #For test shuffle must be false
    pimg = get_prediction(net,test_iter,window_size,stride,patch_size, test_w, test_h, device)
    pimg8 = img_float_to_uint8(pimg)
    pimg8_L = Image.fromarray(pimg8, 'L')
    pimg8_L.save(prediction_test_dir + "prediction_" + str(i) + ".png")
    
# Get prediction file for submission
generate_submission()