# Pipeline of AlexNet, from image processing to prediction file for submission
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

# AlexNet model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            nn.Dropout(0.25), 
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
# Initialize parameters for training
net = AlexNet().float()
lr, num_epochs = 0.0001, 30
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
    # Generate patches for prediction, for traditional CNN resnet should set to 'False'
    testset = TestPatchedDataset(image_filename, window_size, patch_size, stride, resnet=False) 
    # For test dataloader shuffle must be false to follow the submission order
    test_iter = utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    pimg = get_prediction(net,test_iter,window_size,stride,patch_size, test_w, test_h, device)
    pimg8 = img_float_to_uint8(pimg)
    pimg8_L = Image.fromarray(pimg8, 'L')
    pimg8_L.save(prediction_test_dir + "prediction_" + str(i) + ".png")
    
# Get prediction file for submission
generate_submission()