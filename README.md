# Multimodal deep learning from satellite and street-level imagery for measuring income, overcrowding, and environmental deprivation in urban areas
## Esra Suel, Samir Bhatt, Michael Brauer, Seth Flaxman and Majid Ezzati <br> Remote Sensing of Environment, Volume 257, May 2021
This repository holds the code used to implement the proposed methods and produce the experimental results in our article. Unfortunately, we cannot share the data nor the trained models at this moment due to regulatory constraints. Instead, we aim to describe the curated data and provide command line arguments for training and testing the models. For detailed information on the method and the experiments, please refer to the [paper](https://www.sciencedirect.com/science/article/pii/S0034425721000572). 

### Dataset
The dataset consists of street-level images and satellite images from the same geographical location, longitude and lattitude of the street level images, and labels associated with both street-level and satellite images. 

The lattitude and longitude of street level images are necessary to establish correspondence between street level images and pixels in the satellite image. 

In the model, we do not directly use the street-level images because the number of images is not very high to learn a deep model. Instead, we used a VGG16's convolutional layers to convert a street-level image into a 2048 dimensional representation. This representation is used in the model presented here. There is no reason not to adapt a new network to take as input the images directly. One can create such a network and use it in this repository as well. 

The code in this repository requires hdf5 files. There are two sets of hdf5 files: 
1. HDF5 file holding street-level image representations. For each location, there are 4 street-view images in different directions. Each image is represented with a 2048 dimensional vector obtained using the convolutional layers of a VGG16 as further described in the paper. 
2. HDF5 file holding the satellite images. 

