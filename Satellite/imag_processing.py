# Helper functions for basic image processing
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from scipy import ndimage
import torch
import random
import re



def load_image(infilename):
    """Load image """
    data = mpimg.imread(infilename)
    return data


def load_img(root_dir, num_traset):
    """Load image and convert it into the form of numpy array given image path and number of images will be loaded """
    image_dir = root_dir + "/images/"
    files = os.listdir(image_dir)
    n = min(num_traset,len(files))
    print("Loading " + str(n) + " images")
    imgs = np.asarray([load_image(image_dir + files[i]) for i in range(n)])
    print(files[0])
    gt_dir = root_dir + "/groundtruth/"
    print("Loading " + str(n) + " images")
    gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(n)])
    print(files[0])

    return imgs, gt_imgs

def img_float_to_uint8(img):
    """Convert image in numpy array form into uint8 format """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    """Concatenate an image and its groundtruth"""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def label_to_img(imgwidth, imgheight, w, h, labels):
    """Convert array of labels to an prediction image with the form of numpy array """
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """
    Generate overlay images given aerial image and predicted image in form of numpy array.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def extract_features(img):
    """Extract 6-dimensional features consisting of average RGB color as well as variance"""
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def extract_features_2d(img):
    """Extract 2-dimensional features consisting of average gray color as well as variance"""
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

def extract_img_features(filename):
    """Extract features for a given image"""
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def value_to_class(v):
    """Assign labels due to threshold given predicted value """
   # percentage of pixels > 1 required to assign a foreground label to a patch
    foreground_threshold = 0.25
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def value_to_class_ts(v):
    """Assign labels due to threshold given predicted value in tensor form """
    foreground_threshold = 0.25
    df = torch.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def get_rotated_images(images, angles):
    """
    Rotate images with given angles.
    images: list of numpy arrays of the images
    angles: angle list to rotation

    return:
    rotated_images: list of numpy arrays of rotated images
    """
    rotated_images = [None]*(len(images)*len(angles))
    i = 0
    for angle in angles:
        for image in images:
            rotated_images[i] = ndimage.rotate(image, angle, mode='reflect', order=0, reshape=False)
            i += 1
    return rotated_images

def get_flipped_images(images, direction):
    """
    images: list of numpy arrays of the images
    direction: flip horizontally if direction == 0, flip vertically if direction == 1


    return:
    flipped_images: list of numpy arrays of flipped images
    """
    flipped_images = [None]*(len(images))
    if direction == 0:
        for i, img in enumerate(images):
            flipped_images[i] = np.fliplr(img)
    if direction == 1:
        for i, img in enumerate(images):
            flipped_images[i] = np.flipud(img)

    return flipped_images

def get_shifted_images(images):
    """
    images: list of numpy arrays of the images

    return:
    shifted_images: list of numpy arrays of shifted images with a ratio 0.25 of the width horizontally and vertically
    """
    w = images[0].shape[0]
    h = images[0].shape[1]
    # shift = np.array([int(0.25*w), int(0.25*h), 0])
    if len(images[0].shape) > 2:
        shift = [int(0.25*w), int(0.25*h), 0]
    else:
        shift = [int(0.25*w), int(0.25*h)]
    shifted_images = [None]*(len(images))
    for i, img in enumerate(images):
        shifted_images[i] = ndimage.shift(img, shift, mode='reflect')

    return shifted_images



def pad_image(data, padding):
    """Pad image with the reflective strategy. Mirror boundary conditions are applied."""
    if len(data.shape) < 3:
        # Greyscale image (ground truth)
        data = np.lib.pad(data, ((padding, padding), (padding, padding)), 'reflect')
    else:
        # RGB image
        data = np.lib.pad(data, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    return data


def img_crop_stride(im, w, h, stride):
    """Crop image into patches with specified width and height using given stride """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight-h+1,stride):
        for j in range(0,imgwidth-w+1,stride):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches






