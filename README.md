# Multimodal deep learning from satellite and street-level imagery for measuring income, overcrowding, and environmental deprivation in urban areas
## Esra Suel, Samir Bhatt, Michael Brauer, Seth Flaxman and Majid Ezzati <br> Remote Sensing of Environment, Volume 257, May 2021
This repository holds the code used to implement the proposed methods and produce the experimental results in our article. Unfortunately, we cannot share the data nor the trained models at this moment due to regulatory constraints. Instead, we aim to describe the curated data and provide command line arguments for training and testing the models. For detailed information on the method and the experiments, please refer to the [paper](https://www.sciencedirect.com/science/article/pii/S0034425721000572). 

### Dataset
The dataset consists of street-level images and satellite images from the same geographical location, longitude and lattitude of the street level images, and labels associated with both street-level and satellite images. 

The lattitude and longitude of street level images are necessary to establish correspondence between street level images and pixels in the satellite image. 

In the model, we do not directly use the street-level images because the number of images is not very high to learn a deep model. Instead, we used a VGG16's convolutional layers to convert a street-level image into a 2048 dimensional representation. This representation is used in the model presented here. There is no reason not to adapt a new network to take as input the images directly. One can create such a network and use it in this repository as well. 

The code in this repository requires hdf5 files. There are three hdf5 files: 
1. **Street-level HDF5:** HDF5 file holding street-level image representations. For each location, there are 4 street-view images in different directions. Each image is represented with a 2048 dimensional vector obtained using the convolutional layers of a VGG16 as further described in the paper. The HDF5 holds an array of dimension Nx4x2048, with N indicating the number of street-level images in total. 
2. **Satellite HDF5:** HDF5 file holding the satellite images. The satellite images are converted into an array of non-overlapping 1000x1000 patches. Each patch consists of 4 channels, the bands that were included in the images we used. A fifth channel is appended to indicate the correspondence to street-level images. In this fifth channels, either a pixel has the ID of the street level image that corresponds to that location (computed through longitude and lattitude) or -1, which indicates no street-level image is available at that location. As the ID, we used the row number of the street level image in the street-level HDF5 file (row number corresponds to the first index). The HDF5 holds an array of dimension Mx1000x1000x5, where M is the number of patches.
3. **Satellite labels:** Labels corresponding to satellite images are pixel-wise labels. They have the same size as the satellite images. Like the satellite images, we divide these images into 1000x1000 non-overlapping patches corresponding to the satellite HDF5 file. Labels are categorical but saved as integers. 

Labels are different for the street-level dataset than satellite images. Street-level labels are saved in a CSV file. The rows of the CSV files correspond to the first index of the array stored in the street-level HDF5. 

### Example training and testing command lines
