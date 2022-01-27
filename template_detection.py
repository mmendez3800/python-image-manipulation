"""
This program detects if a template image can be found within a source image
It equates to whether one image can be found as an instance (or multiple) within
    a second image
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

"""
Perform suppression on a matrix by zeroing one the neighbors of
the pixel (based on the method used)

Parameters
----------
matrix : numpy.array (dtype = float)
    an HXW array containing score values from previous template matching
    
method : one of {'cc','ncc','ssd'}
    Method for comparing template and image patch
    
Returns
-------
numpy.array (dtype = float)
    an HXW array where neighbors have been suppressed (based on method)
"""
def suppress_values(matrix, method):
    
    H, W = np.shape(matrix)
    
    #Created new matrix to be used to retain desired min/max values
    if method == 'ssd':
        suppressed = np.full((H, W), fill_value = H * W, dtype = 'float')
    elif method == 'cc' or method == 'ncc':
        suppressed = np.zeros((H, W), dtype = 'float')
    else:
        print("Invalid Method Entered. Please Enter Valid Method.")
        return None
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if method == 'ssd':
                
                #Checks that matrix value is smallest of surrounding neighbors
                #If true, retains value in resulting matrix
                if matrix[i, j] <= matrix[i - 1, j - 1] and matrix[i, j] <= matrix[i - 1, j] and \
                matrix[i, j] <= matrix[i - 1, j + 1] and matrix[i, j] <= matrix[i, j - 1] and \
                matrix[i, j] <= matrix[i, j + 1] and matrix[i, j] <= matrix[i + 1, j - 1] and \
                matrix[i, j] <= matrix[i + 1, j] and matrix[i, j] <= matrix[i + 1, j + 1]:
                    suppressed[i, j] = matrix[i, j]
            else:
                
                #Checks that matrix value is largest of surrounding neighbors
                #If true, retains value in resulting matrix
                if matrix[i, j] >= matrix[i - 1, j - 1] and matrix[i, j] >= matrix[i - 1, j] and \
                matrix[i, j] >= matrix[i - 1, j + 1] and matrix[i, j] >= matrix[i, j - 1] and \
                matrix[i, j] >= matrix[i, j + 1] and matrix[i, j] >= matrix[i + 1, j - 1] and \
                matrix[i, j] >= matrix[i + 1, j] and matrix[i, j] >= matrix[i + 1, j + 1]:
                    suppressed[i, j] = matrix[i, j]
    
    return suppressed   

"""
Detect a object in an image by performing cross-correlation with the
supplied template

image : numpy array (dtype = float)
    An 2D array containing pixel brightness values of image

template : numpy array (dtype = float)
    Template we wish to match
  
thresh : float
    Score threshold above which we declare a positive match

method : one of {'cc','ncc','ssd'}
    Method for comparing template and image patch

Returns
-------
numpy array (dtype = float)
    output of utilizing method with image section and template (prior to threshold)

numpy array (dtype = float)
    detections of template matchings in image
    represented by indices where detection occurred
"""
def detect_template(image, template, thresh, method):
    img = image
    temp = template
    
    #Dimensions of image and temiplate
    imgH, imgW = np.shape(image)
    tempH, tempW = np.shape(template)
    
    result = np.zeros(((imgH - tempH), (imgW - tempW)), dtype = 'float')
    
    #Updates image and template by subtracting from their mean value
    if method == 'cc' or method == 'ncc':
        img = img - np.mean(img)
        temp = temp - np.mean(temp)
    
    for i in range(0, (imgH - tempH)):
        for j in range (0, (imgW - tempW)):
            
            #Section of the image being used for comparison with template
            section = img[i:(i + tempH), j:(j + tempW)]
            
            #Cross Correlation calculation
            if method == 'cc':
                result[i, j] = np.sum(section * temp)
                
            #Normalized Cross Correlation calculation
            elif method == 'ncc':
                num = np.sum(section * temp)
                denom = np.sqrt(np.sum(section ** 2) * np.sum(temp ** 2))
                result[i, j] = num / denom
            
            #Sum Of Squared Differences Calculation
            elif method == 'ssd':
                result[i, j] = np.sum((section - temp) ** 2)
            
            else:
                print("Invalid Method Entered. Please Enter Valid Method.")
                return None
    
    #Suppresses values of resulting template matching
    #Checks for indices of matrix that meet threshold given
    suppressed = suppress_values(result, method)
    if method == 'ssd':
        suppressed = np.argwhere(suppressed <= thresh)
    else:
        suppressed = np.argwhere(suppressed >= thresh)
    
    return result, suppressed

#Conversion of image and template to grayscale
img = Image.open(os.getcwd() + '/dilbert1.jpg').convert('L')
img = np.array(img, dtype = 'float') / 256
temp = Image.open(os.getcwd() + '/template.jpg').convert('L')
temp = np.array(temp, dtype = 'float') / 256

#Obtain cross correlation output and detection results
visual1, result1 = detect_template(img, temp, 20, 'cc')
visual2, result2 = detect_template(img, temp, 0.8, 'ncc')
visual3, result3 = detect_template(img, temp, 10, 'ssd')

tempH, tempW = np.shape(temp)

#Display results
fig, axs = plt.subplots(4, 2, figsize = (15, 15))

axs[0][0].set_title("Image")
axs[0][0].imshow(img, cmap = 'gray')
axs[0][1].set_title("Template")
axs[0][1].imshow(temp, cmap = 'gray')

axs[1][0].set_title("CC - Visualization")
im = axs[1][0].imshow(visual1, cmap = 'jet')
fig.colorbar(im, ax = axs[1][0])
axs[1][1].set_title("CC - Detection")
axs[1][1].imshow(img, cmap = 'gray')
for i in range(0, np.shape(result1)[0]):
    w = result1[i][1]
    h = result1[i][0]
    rect = patches.Rectangle((w, h), tempW, tempH, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    axs[1][1].add_patch(rect)

axs[2][0].set_title("NCC - Visualization")
im = axs[2][0].imshow(visual2, cmap = 'jet')
fig.colorbar(im, ax = axs[2][0])
axs[2][1].set_title("NCC - Detection")
axs[2][1].imshow(img, cmap = 'gray')
for i in range(0, np.shape(result2)[0]):
    w = result2[i][1]
    h = result2[i][0]
    rect = patches.Rectangle((w, h), tempW, tempH, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    axs[2][1].add_patch(rect)
    
axs[3][0].set_title("SSD - Visualization")
im = axs[3][0].imshow(visual3, cmap = 'jet')
fig.colorbar(im, ax = axs[3][0])
axs[3][1].set_title("SSD - Detection")
axs[3][1].imshow(img, cmap = 'gray')
for i in range(0, np.shape(result3)[0]):
    w = result3[i][1]
    h = result3[i][0]
    rect = patches.Rectangle((w, h), tempW, tempH, linewidth = 2, edgecolor = 'red', facecolor = 'none')
    axs[3][1].add_patch(rect)

plt.show()
