# Pytorch dataset in the form of patch
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from imag_processing import *
from patch_prepare import *
from torchvision import transforms

class TrainPatchedDataset(Dataset):
    """
 Generate patched form with target window size and stride dataset for training and validation with pre-transformation.
    """
    def __init__(self, root_dir, num_traset, indices_tra, window_size, patch_size, stride, rotate, flip, shift, resnet):
        self.resnet = resnet
        # Load images and labels
        patches_tra_np, labels_tra_np = prepare_train_patches(root_dir, num_traset, indices_tra, window_size, 
                                                          patch_size, stride, rotate, flip, shift)
        self.patched_images = patches_tra_np
        self.patched_labels = labels_tra_np
        
        # Define out transformations for ResNet pretrained on ImageNet dataset, including normalization and resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_patches_transform_res = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        self.normalize
    ])
        # Define out transformations for traditional CNN models
        self.train_patches_transform = transforms.ToTensor()
        #self.train_labels_transform = transforms.ToTensor()
        #self.images_transform = torch.from_numpy

    def __getitem__(self, index):
        patched_image = self.patched_images[index]
        patched_label = self.patched_labels[index]

        if self.resnet == True:
            patched_image_uint8 = img_float_to_uint8(patched_image)
            patched_image_PIL = Image.fromarray(patched_image_uint8.astype('uint8'), 'RGB')
            patched_image = self.train_patches_transform_res(patched_image_PIL)
            patched_label = torch.tensor(patched_label)
        else:
            #patched_image = np.transpose(self.patched_images[index], (2, 0, 1))
            patched_image = self.train_patches_transform(patched_image)
            patched_label = torch.tensor(patched_label)
            
        return patched_image.float(), patched_label.long()

    def __len__(self):
        return len(self.patched_images)
    
    
    
    
    
class TestPatchedDataset(Dataset):
    """
 Generate patched form dataset with target window size and stride for test with pre-transformation.
    """
    def __init__(self, image_filename, window_size, patch_size, stride, resnet):
        self.resnet = resnet
        # Load images and labels
        patches_test_np = prepare_test_patches(image_filename, window_size, 
                                                          patch_size, stride)
        self.patched_images = patches_test_np
        
        # Define out transformations for ResNet pretrained on ImageNet dataset, including normalization and resize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.test_patches_transform_res = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        self.normalize
    ])
        # Define out transformations for traditional CNN models
        self.test_patches_transform = transforms.ToTensor()


    def __getitem__(self, index):
        patched_image = self.patched_images[index]
        
        if self.resnet == True:
            patched_image_uint8 = img_float_to_uint8(patched_image)
            patched_image_PIL = Image.fromarray(patched_image_uint8.astype('uint8'), 'RGB')
            patched_image = self.test_patches_transform_res(patched_image_PIL)
        else:
            patched_image = self.test_patches_transform(patched_image)
            
        return patched_image.float()

    def __len__(self):
        return len(self.patched_images)