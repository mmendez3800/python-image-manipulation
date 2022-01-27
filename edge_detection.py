"""
This program detects the edges in an image
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image
from scipy import ndimage

"""
Compute the gradient orientation and magnitude after smoothing a
grayscale image with a Gaussian filter with parameter sigma

Parameters
----------
gray : numpy.array (dtype = float)
    a HxW array of a grayscale image

sig_val: float
    the width of the Gaussian filter
    
Returns
-------
tuple of numpy.array (dtype = float)
    two HxW arrays containing the gradient orientation and magnitude
"""
def image_gradient(gray, sig_val):
    
    #Compute horizontal (x) derivative and vertical (y) derivate of image
    #Done by convolving derivative of Gaussian filter from x and y axis with image
    y_deriv = ndimage.gaussian_filter(gray, sigma = sig_val, order = [1, 0])
    x_deriv = ndimage.gaussian_filter(gray, sigma = sig_val, order = [0, 1])
    
    #Caclulate magnitude and orientation of gradient
    magnitude = np.hypot(x_deriv, y_deriv)
    orientation = np.arctan2(y_deriv, x_deriv)
    
    return magnitude, orientation

"""
Perform non-maximum suppression on a gradient magnitude using the
corresponding orientation values of the pixels

Parameters
----------
mag : numpy.array (dtype = float)
    the gradient magnitude
    
ang : numpy.array (dtype = float)
    the gradient orientation
    
Returns
-------
numpy.array (dtype = float)
    suppressed version of gradient magnitude
"""
def non_maximum_suppress(mag, ang):
    
    import math
    
    #Create numpy array to represent compressed version of gradient magnitude
    H, W = np.shape(mag)
    suppressed = np.zeros((H, W), dtype = 'float')
    
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            
            #Update negative radian values from gradient orientation to their respective positive radian values
            if ang[i][j] < 0:
                ang[i][j] += 2 * math.pi
            
            #Checks if orientation is within the boundaries of 0° or 180°
            if (ang[i][j] >= (15 * math.pi / 8) or ang[i][j] < (math.pi / 8)) or \
            ((9 * math.pi / 8) > ang[i][j] >= (7 * math.pi / 8)):
                
                #Checks if current pixel is maximum of surrounding pixels
                #If true, then value is stored in suppressed matrix
                if mag[i][j] >= mag[i][j - 1] and mag[i][j] >= mag[i][j + 1]:
                    suppressed[i][j] = mag[i][j]
            
            #Checks if orientation is within the boundaries of 45° or 225°
            elif ((3 * math.pi / 8) > ang[i][j] >= (math.pi / 8)) or \
            ((11 * math.pi / 8) > ang[i][j] >= (9 * math.pi / 8)):
                
                #Checks if current pixel is maximum of surrounding pixels
                #If true, then value is stored in suppressed matrix
                if mag[i][j] >= mag[i + 1][j - 1] and mag[i][j] >= mag[i - 1][j + 1]:
                    suppressed[i][j] = mag[i][j]
            
            #Checks if orientation is within the boundaries of 90° or 270°
            elif ((5 * math.pi / 8) > ang[i][j] >= (3 * math.pi / 8)) or \
            ((13 * math.pi / 8) > ang[i][j] >= (11 * math.pi / 8)):
                
                #Checks if current pixel is maximum of surrounding pixels
                #If true, then value is stored in suppressed matrix
                if mag[i][j] >= mag[i - 1][j] and mag[i][j] >= mag[i + 1][j]:
                    suppressed[i][j] = mag[i][j]
            
            #Checks if orientation is within the boundaries of 135° or 315°
            elif ((7 * math.pi / 8) > ang[i][j] >= (5 * math.pi / 8)) or \
            ((15 * math.pi / 8) > ang[i][j] >= (13 * math.pi / 8)):
                
                #Checks if current pixel is maximum of surrounding pixels
                #If true, then value is stored in suppressed matrix
                if mag[i][j] >= mag[i - 1][j - 1] and mag[i][j] >= mag[i + 1][j + 1]:
                    suppressed[i][j] = mag[i][j]
    
    return suppressed
    
"""
Detect edges in an image using the given smoothing and threshold parameters.

Parameters
----------
gray : numpy.array (dtype = float)
    a HxW array of a grayscale image

sig_val : float
    the width of the Gaussian filter

thresh : float
    the threshold value for determining if a pixel is part of an edge or not
    
Returns
-------
numpy.array (dtype = float)
    image edges outline based on imput image and threshold provided
"""
def detect_edge(gray, sig_val, thresh):
    
    #Calculate gradient magnitude and gradient orientation
    magnitude, orientation = image_gradient(gray, sig_val)
    suppressed = non_maximum_suppress(magnitude, orientation)
    
    #Create numpy array to represent compressed version of gradient magnitude
    H, W = np.shape(suppressed)
    edges = np.zeros((H, W), dtype = 'float')
    
    for i in range(0, H):
        for j in range(0, W):
            
            #Checks if pixel passes threshold value
            #If true, pixel is treated as an edge
            if suppressed[i][j] >= thresh:
                edges[i][j] = 255
        
    return edges

#Synthetic test image for debugging purposes
[yy, xx] = np.mgrid[-100:100, -100:100]
img1 = np.minimum(np.maximum(np.array(xx * xx + yy * yy, dtype=float), 400), 8100)

#Second image opened and converted to grayscale
img2 = Image.open(os.getcwd() + '/house.jpg').convert('L')
img2 = np.array(img2, dtype = 'float')

#Gradient magnitude, gradient orientation, and edges obtained
#Done for different values of sigmas with Gaussian filter
mag11, ang11 = image_gradient(img1, 1)
edge11 = detect_edge(img1, 1, 20)
mag12, ang12 = image_gradient(img1, 1.5)
edge12 = detect_edge(img1, 1.5, 20)
mag21, ang21 = image_gradient(img2, 1)
edge21 = detect_edge(img2, 1, 20)
mag22, ang22 = image_gradient(img2, 1.5)
edge22 = detect_edge(img2, 1.5, 20)

#Display results
fig, axs = plt.subplots(4, 4, figsize = (17, 17))

axs[0][0].set_title("Image 1")
axs[0][0].imshow(img1, cmap = 'gray')
axs[0][1].set_title("Magnitude (\u03C3 = 1)")
im = axs[0][1].imshow(mag11, cmap = 'gray')
fig.colorbar(im, ax = axs[0][1])
axs[0][2].set_title("Angle (\u03C3 = 1)")
im = axs[0][2].imshow(ang11, cmap = 'hsv')
fig.colorbar(im, ax = axs[0][2])
axs[0][3].set_title("Edges")
axs[0][3].imshow(edge11, cmap = 'gray')

axs[1][0].set_title("Image 1")
axs[1][0].imshow(img1, cmap = 'gray')
axs[1][1].set_title("Magnitude (\u03C3 = 1.5)")
im = axs[1][1].imshow(mag12, cmap = 'gray')
fig.colorbar(im, ax = axs[1][1])
axs[1][2].set_title("Angle (\u03C3 = 1.5)")
im = axs[1][2].imshow(ang12, cmap = 'hsv')
fig.colorbar(im, ax = axs[1][2])
axs[1][3].set_title("Edges")
axs[1][3].imshow(edge12, cmap = 'gray')

axs[2][0].set_title("Image 2")
axs[2][0].imshow(img2, cmap = 'gray')
axs[2][1].set_title("Magnitude (\u03C3 = 1)")
im = axs[2][1].imshow(mag21, cmap = 'gray')
fig.colorbar(im, ax = axs[2][1])
axs[2][2].set_title("Angle (\u03C3 = 1)")
im = axs[2][2].imshow(ang21, cmap = 'hsv')
fig.colorbar(im, ax = axs[2][2])
axs[2][3].set_title("Edges")
axs[2][3].imshow(edge21, cmap = 'gray')

axs[3][0].set_title("Image 2")
axs[3][0].imshow(img2, cmap = 'gray')
axs[3][1].set_title("Magnitude (\u03C3 = 1.5)")
im = axs[3][1].imshow(mag22, cmap = 'gray')
fig.colorbar(im, ax = axs[3][1])
axs[3][2].set_title("Angle (\u03C3 = 1.5)")
im = axs[3][2].imshow(ang22, cmap = 'hsv')
fig.colorbar(im, ax = axs[3][2])
axs[3][3].set_title("Edges")
axs[3][3].imshow(edge22, cmap = 'gray')

plt.show()
