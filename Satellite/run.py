# Generate the same .csv predictions of our best submission to the competition on AIcrowd(F1score:0.891)
# A submission file named 'submission_best.csv' will be generated in the folder 'submission' when you run this file, prediction 
# image corresponding to the best submission will be created in the folder 'predictions_test' :-)
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

# Initialize parameters
root_dir = "Datasets/training"
num_traset = 100
window_size = 72
patch_size = 16
stride = 16
batch_size = 64
num_workers=0
ratio_tra = 0
ratio_test = 1-ratio_tra

# Load pretrained model
Resnet34_PT = models.resnet34(pretrained=True)
Resnet34_PT.fc = nn.Linear(512, 2)
print(Resnet34_PT.fc)
net = Resnet34_PT.float()

# Load parameters of the best trained model with 'ResNet_best_gpu.pt' if you use gpu
# Load parameters of the best trained model with 'ResNet_best_cpu.pt' if you use cpu
PATH = "./model_para/ResNet_best_gpu.pt"
#PATH = "./model_para/ResNet_best_cpu.pt" #Load this parameter if you use cpu, then the run time can be very long.
net.load_state_dict(torch.load(PATH))


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
    # Generate patches for prediction, for ResNet resnet should set to 'True'
    testset = TestPatchedDataset(image_filename, window_size, patch_size, stride, resnet=True)
    test_iter = utils.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers) #For test shuffle must be false
    pimg = get_prediction(net,test_iter,window_size,stride,patch_size, test_w, test_h, device)
    pimg8 = img_float_to_uint8(pimg)
    pimg8_L = Image.fromarray(pimg8, 'L')
    pimg8_L.save(prediction_test_dir + "prediction_" + str(i) + ".png")

# Generate submission file
generate_submission()