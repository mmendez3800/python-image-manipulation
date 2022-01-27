"""
This program creates a hybrid of two images
It uses a simple version of the SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import gaussian_filter

"""
Given an image, split the image into two frequency bands
and return the two components (low, high)

Parameters
----------
image : numpy.array (dtype = uint8)
    a HxWx3 array of the image

sigma : float
    the width of the gaussian filter
    
Returns
-------
tuple of numpy.array (dtype = float)
    two HxWx3 arrays containing low/high freq components respectively
"""
def split_bands(image, sigma_val):
    image_data = image.astype(float) / 256
    
    #Separate RGB channels of image
    image_r = image_data[:, :, 0]
    image_g = image_data[:, :, 1]
    image_b = image_data[:, :, 2]
    
    #Obtain low frequency component of each color chanel
    low_r = gaussian_filter(image_r, sigma = sigma_val)
    low_g = gaussian_filter(image_g, sigma = sigma_val)
    low_b = gaussian_filter(image_b, sigma = sigma_val)
    
    #Obtain high frequency component of each color chanel
    high_r = np.subtract(image_r, low_r)
    high_g = np.subtract(image_g, low_g)
    high_b = np.subtract(image_g, low_b)
    
    #Combine all components into their respective arrays
    low_freq = np.dstack((low_r, low_g, low_b))
    high_freq = np.dstack((high_r, high_g, high_b))
    
    return low_freq, high_freq

"""
Given the frequency components of an image, converts the components to grayscale

Parameters
----------
low_freq : numpy.array (dtype = float)
    a HxWx3 array of the low frequency component of an image

high_freq: numpy.array (dtype = float)
    a HxWx3 array of the high frequency component of an image
    
Returns
-------
tuple of numpy.array (dtype =float)
    two HxW arrays containing low/high freq components respectively in grayscale
"""
def convert_grayscale(low_freq, high_freq):
    low_freq_gray = np.dot(low_freq[:, :, 0:3], [0.2898, 0.5870, 0.1140])
    high_freq_gray = np.dot(high_freq[:, :, 0:3], [0.2898, 0.5870, 0.1140])
    
    return low_freq_gray, high_freq_gray

#Obtain images for testing
file_path_1 = os.getcwd() + "/rhino_resize.jpg"
image_data_1 = plt.imread(file_path_1)
file_path_2 = os.getcwd() + "/lion_resize.jpg"
image_data_2 = plt.imread(file_path_2)

#Calculate low frequency component and high frequency component of images
low_freq_data_1, high_freq_data_1 = split_bands(image_data_2, 6)
low_freq_data_2, high_freq_data_2 = split_bands(image_data_1, 10)
low_freq_data_3, high_freq_data_3 = split_bands(image_data_1, 2.5)
low_freq_data_4, high_freq_data_4 = split_bands(image_data_2, 8)

#Convert components to grayscale for display
low_freq_gray_1, high_freq_gray_1 = convert_grayscale(low_freq_data_1, high_freq_data_1)
low_freq_gray_2, high_freq_gray_2 = convert_grayscale(low_freq_data_2, high_freq_data_2)
low_freq_gray_3, high_freq_gray_3 = convert_grayscale(low_freq_data_3, high_freq_data_3)
low_freq_gray_4, high_freq_gray_4 = convert_grayscale(low_freq_data_4, high_freq_data_4)

#Create hybrid image by combining frequency components
hybrid_1 = np.add(low_freq_gray_2, high_freq_gray_1)
hybrid_2 = np.add(low_freq_gray_4, high_freq_gray_3)

#Display input images and hybrid images
fig, axs = plt.subplots(2, 2, figsize = (10, 8))
axs[0][0].imshow(image_data_1)
axs[0][0].set_title("Input Image 1")
axs[0][1].imshow(image_data_2)
axs[0][1].set_title("Input Image 2")
axs[1][0].imshow(hybrid_1, cmap = 'gray')
axs[1][0].set_title("Hybrid Image 1")
axs[1][1].imshow(hybrid_2, cmap = 'gray')
axs[1][1].set_title("Hybrid Image 2")
plt.show()