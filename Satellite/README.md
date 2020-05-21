# Road Segmentation on Satellite Images Using Convolutional Neural Networks

**Team Members: Danya Li, Shichao Jia, Zhuoyue Wang**

## Requirements

- Python 3
- NumPy
- Pillow
- SciPy
- Random
- PyTorch
- TorchVision
- matplotlib
- scikit-learn

## Dataset

### Training

Training images and corresponding ground-truth images should be stored in the folder ```./Datasets/training/images``` and ```./Datasets/training/groundtruth```. Currently they are removed given the limited file size allowed for submission

### Test

Test images are stored in ```./Datasets/test_set_images```, and every test image is stored in a single folder named ```test_*``` where ```*``` is the index of the image. Currently they are removed given the limited file size allowed for submission.

## Code

- ```patch_prepare.py```
  - Implement augmentation on the training data set and slice the augmented images with a given patch size and a given stride
- ```dataset.py```
  - Generate the data sets of the training images and the test images, respectively, in the form of patches
- ```imag_processing.py	```
  - Contain all the helper functions regarding image manipulation and processing

- ```pred.py```
  - Give the predictions of a single image in the ```numpy.array``` data type based on a given model

- ```LeNet_base.py```
  - Pipeline of baseline model LeNet, from image processing to prediction file for submission
- ```LeNet_modified.py```
  - Pipeline of modified LeNet, from image processing to prediction file for submission
- ```AlexNet.py```
  - Pipeline of AlexNet, from image processing to prediction file for submission
- ```ResNet.py```
  - Pipeline of ResNet model, from image processing to prediction file for submission
- ```train.py```
  - Training functions for all the models
- ```generate_submission.py```
  - Generate a ```.csv``` file according to the segmented images for submission
- ```run.py```
  - Generate the same ``` .csv``` file as our best submission to the challenge on AIcrowd
  - A submission file named ```submission_best.csv``` will be generated in the folder ```./submission``` when you run this file
  - Results of segmentation on the test images corresponding to the best submission will be created in the folder ```./predictions_test``` :-)

## Prediction

The folder ```./predictions_test_backup``` contains all the results of segmentaion on the test images

## Model Parameters

The files ```ResNet_best_cpu.pt``` and  ```ResNet_best_gpu.pt``` in the folder ```./model_para```  contain all the parameters in the ResNet model running on cpu and gpu, respectively.