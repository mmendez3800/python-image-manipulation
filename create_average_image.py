"""
This program creates an average image using a set of images provided
A folder is referenced to display the results based on two image sets
"""

import matplotlib.pyplot as plt
import numpy as np
import os

"""
Computes the average of all color images in a specified directory and returns the result.

Parameters
----------
dirname : str
    Directory to search for images
    Assumes that directory is within same path as Jupyter Python notebook
      
Returns
-------
numpy.array (dtype=float)
    HxWx3 array containing the average of the images found
"""
def average_image(dirname):
    
    from statistics import mean
    from PIL import Image
    
    #Opens the list of files within directory
    path = os.getcwd() + dirname
    dir_items = os.listdir(path)
    
    #Lists to retain the heights and widths of images
    image_heights = []
    image_widths = []
    
    for file in dir_items:
        try:
            image = plt.imread(path + file)
            
            #Checks if image is not a colored image or is not a uint8 data type
            if image.ndim != 3 or image.dtype != np.uint8:
                dir_items.remove(file)
                continue
            
            #Adds height and width of valid images to list
            image_heights.append(np.size(image, 0))
            image_widths.append(np.size(image, 1))
        
        except OSError:
            dir_items.remove(file)
            continue
    
    #Computes number of valid images in directory
    #Computes average height and width of images
    num_images = len(dir_items)
    avg_image_height = round(mean(image_heights))
    avg_image_width = round(mean(image_widths))
    
    #Numpy array used to hold the information of our average image
    avg_image = np.zeros((avg_image_height, avg_image_width, 3), dtype = float)
    
    for file in dir_items:
        image = Image.open(path + file)
        resized_image = image.resize((avg_image_width, avg_image_height))
        resized_array = np.array(resized_image)
        
        #Computes average image from resized image
        resized_array = resized_array.astype(float) / 256
        avg_image = avg_image + resized_array / num_images
        
        image.close()
    
    return avg_image

"""
Checks all files within the specified directory and returns a random colored image from the directory.

Parameters
----------
dirname : str
    Directory to search for images
    Assumes that directory entered is within same working directory as Jupyter Python notebook
      
Returns
-------
numpy.array (dtype = uint8)
    HxWx3 array representing the randomly selected image
"""
def random_image(dirname):
    
    import random
    
    #Opens the list of files within directory
    path = os.getcwd() + dirname
    dir_items = os.listdir(path)
    
    #List to retain valid images
    valid_images = []
    
    for file in dir_items:
        try:
            image = plt.imread(path + file)
            
            #Checks if image is not a colored image or is not a uint8 data type
            if image.ndim != 3 or image.dtype != np.uint8:
                continue
                
            #Adds file name to list of valid images
            valid_images.append(file)
        
        except OSError:
            continue
    
    return random.choice(valid_images)

#Paths to images from set1 and set2
path1 = "/averageimage_data/set1/"
path2 = "/averageimage_data/set2/"

#Select random image from both paths and open image
rand_image_set1 = random_image(path1)
image_set1 = plt.imread(os.getcwd() + path1 + rand_image_set1)
rand_image_set2 = random_image(path2)
image_set2 = plt.imread(os.getcwd() + path2 + rand_image_set2)

#Compute average image from set1 and set2 in numpy array form
avg_image_set1 = average_image(path1)
avg_image_set2 = average_image(path2)

#Resulting random imgage and average image from each set
fig, axs = plt.subplots(2, 2, figsize = (10, 7))
axs[0][0].imshow(image_set1)
axs[0][0].set_title("Random Image (Set1)")
axs[0][1].imshow(avg_image_set1)
axs[0][1].set_title("Average Image (Set1)")
axs[1][0].imshow(image_set2)
axs[1][0].set_title("Random Image (Set2)")
axs[1][1].imshow(avg_image_set2)
axs[1][1].set_title("Average Image(Set2)")
plt.show()