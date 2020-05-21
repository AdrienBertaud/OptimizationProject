# Implement augmentation on the training and test data set and slice the augmented images with a given patch size and a given stride
from imag_processing import *

def prepare_train_patches(root_dir, num_traset, indices_tra, window_size, patch_size, stride, rotate, flip, shift):
    """Implement augmentation on the training data set and slice the augmented images with a given patch size and a given stride """
    # Loaded a set of images
    #num_traset = 100 #Trainingset numbers

    image_dir = root_dir + "/images/"
    #files = os.listdir(image_dir)
    #n = min(num_traset,len(files)) 
    files = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    imgs = np.asarray([load_image(image_dir + files[i]) for i in indices_tra])
    print("Loading " + str(len(imgs)) + " images")
    gt_dir = root_dir + "/groundtruth/"
    gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in indices_tra])
    print("Loading " + str(len(gt_imgs)) + " groundtruth images")   
    
    imgs_copy = imgs.copy()
    gt_imgs_copy = gt_imgs.copy()
    
    # Initialization of training image information
    padding_size = (window_size - patch_size) // 2
    imag_w = imgs[0].shape[0]
    imag_h = imgs[0].shape[1]
    num_ch = imgs[0].shape[2]
    
    # Transformation
    # rotation
    if rotate == True:
        angles = [45] #Rotation angles
        rotated_imgs = get_rotated_images(imgs_copy, angles)
        gt_rotated_imgs = get_rotated_images(gt_imgs_copy, angles)
        imgs = np.vstack((imgs,rotated_imgs))
        gt_imgs = np.vstack((gt_imgs,gt_rotated_imgs))

    # flip
    if flip == True:
        # flip horizontally
        idx_flip_H = random.sample(range(0, len(imgs_copy)), int(0.25*len(imgs_copy)))
        flipped_imgs_H = get_flipped_images(imgs_copy[idx_flip_H], 0)
        gt_flipped_imgs_H = get_flipped_images(gt_imgs_copy[idx_flip_H], 0)
        imgs = np.vstack((imgs, flipped_imgs_H))
        gt_imgs = np.vstack((gt_imgs, gt_flipped_imgs_H))
        # flip vertically
        idx_flip_V= random.sample(range(0, len(imgs_copy)), int(0.25*len(imgs_copy)))
        flipped_imgs_V = get_flipped_images(imgs_copy[idx_flip_V], 1)
        gt_flipped_imgs_V = get_flipped_images(gt_imgs_copy[idx_flip_V], 1)
        imgs = np.vstack((imgs, flipped_imgs_V))
        gt_imgs = np.vstack((gt_imgs, gt_flipped_imgs_V))

    if shift == True:
        idx_shift = random.sample(range(0, len(imgs_copy)), int(0.25*len(imgs_copy)))
        shifted_images = get_shifted_images(imgs_copy[idx_shift])
        gt_shifted_images = get_shifted_images(gt_imgs_copy[idx_shift])
        imgs = np.vstack((imgs, shifted_images))
        gt_imgs = np.vstack((gt_imgs, gt_shifted_images))

        num_traset_transformation = len(imgs)
    else:
        num_traset_transformation = len(imgs)
        
    print("Procssing " + str(num_traset_transformation) + " images with transformation")  
    
    # Padding training aerial images
    imgs_pad = np.empty((len(imgs),imag_w+2*padding_size, imag_h+2*padding_size, num_ch))
    for i in range(len(imgs)):
        imgs_pad[i] = pad_image(imgs[i], padding_size)
    
    # Crop into patches
    #stride = 16
    img_windows = [img_crop_stride(imgs_pad[i], window_size, window_size, stride) for i in range(num_traset_transformation)]
    gt_patches = [img_crop_stride(gt_imgs[i], patch_size, patch_size, stride) for i in range(num_traset_transformation)]
    
    #Linearize patches
    img_windows_ln = [img_windows[i][j] for i in range(len(img_windows)) for j in range(len(img_windows[i]))]
    gt_patches_ln =  [gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]
    
    # Extract labels for each image patch
    labels_tra_list = [value_to_class(np.mean(gt_patches_ln[i])) for i in range(len(gt_patches_ln))]
    
    #Convert to numpy array
    img_tra_np = np.array(img_windows_ln)
    labels_tra_np = np.array(labels_tra_list)
    print("Loading " + str(len(img_tra_np)) + " patches" + " completed")

    return img_tra_np, labels_tra_np


def prepare_test_patches(image_filename, window_size, patch_size, stride):
    """Implement augmentation on the test data set and slice the augmented images with a given patch size and a given stride"""
    
    img = mpimg.imread(image_filename)
    
    # Initialization of training image information
    padding_size = (window_size - patch_size) // 2
    

    imgs_pad = pad_image(img, padding_size)
    img_test_np = np.asarray(img_crop_stride(imgs_pad, window_size, window_size,stride))
    

    return img_test_np