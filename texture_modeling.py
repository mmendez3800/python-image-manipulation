"""
This program takes an image and filters it with Gaussian derivates
It is done in the horizontal and vertial directs through different sigma scales
There is also an additional center surround filter
This is done by taking the difference of two isotropic Gaussian functions at two different scales
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage

"""
Create requested filterbanks using a specified kernel length

Parameters
----------
length : int
    the requested side length of the kernels being created
    
Returns
-------
tuple of numpy.array (dtype = float)
    a tuple of HXW arrays containing requested filterbanks
"""
def create_filterbanks(length):
    
    """
    Create a Gaussian kernel of specified length and sigma value

    Parameters
    ----------
    size : int
        the requested side length of the kernels being created
    
    sigma : int
        the sigma value of the requested Gaussian kernel

    Returns
    -------
    numpy.array (dtype = float)
        an HXW array representing the Gaussian kernel
    """
    def gaussian_kernel(size, sigma):
        
        #Create arrays based on size input
        values = (size - 1) / 2
        y, x = np.ogrid[-values:(values + 1), -values:(values + 1)]
        
        #Perform calculations to obtain Gaussian kernel
        kernel = np.exp( -(x * x + y * y) / (2 * sigma * sigma) )
        kernel /= kernel.sum()
        
        return kernel
    
    #Gaussian kernels of various sigma values
    gauss1 = gaussian_kernel(length, 1)
    gauss2 = gaussian_kernel(length, 2)
    gauss4 = gaussian_kernel(length, 4)
    
    #Horizontal and vertical derivatives of Gaussian kernels
    deriv1v, deriv1h = np.gradient(gauss1)
    deriv2v, deriv2h = np.gradient(gauss2)
    deriv4v, deriv4h = np.gradient(gauss4)
    
    #Computation of center surround filters
    center1 = gauss2 - gauss1
    center2 = gauss4 - gauss2
    
    return deriv1h, deriv1v, deriv2h, deriv2v, deriv4h, deriv4v, center1, center2

"""
Apply requested filterbanks using a specified kernel length onto grayscale image

Parameters
----------
image : numpy.array (dtype = float)
    a HXW array representing a grayscale image

length : int
    the requested side length of the kernels being created
    
Returns
-------
list of numpy.array (dtype = float)
    a list of HXW arrays containing filterbanks applied onto grayscale image
"""
def apply_filterbanks(image, size):
    
    filterbanks = create_filterbanks(size)
    responses = []
    
    for kernel in filterbanks:
        
        #Apply filterbanks onto grayscale image
        result = ndimage.convolve(image, kernel)
        responses.append(result)
    
    return responses

#Read image and convert to grayscale
image = plt.imread('plant_resize.png')
image = image.astype('float')
grayscale = np.dot(image[:, :, 0:3], [0.2989, 0.5870, 0.1140])

#Apply filterbanks to image
results = apply_filterbanks(grayscale, 9)

#Create subplot to plot response images
fig, ax = plt.subplots(5, 2, figsize=(10, 22))

#Plot original image
ax[0][0].imshow(grayscale, cmap='gray')
ax[0][1].axis('off')

i = 1
j = 0
for index in range(len(results)):
    axs = ax[i][j]
    
    #Plot response images
    kernel = axs.imshow(results[index], cmap='gray')
    
    if (j + 1) != 2:
        j += 1
    else:
        j = 0
        i += 1

#Set titles for each plot
ax[0][0].set_title('Original Image')
ax[1][0].set_title('Horizontal: \u03C3 = 1')
ax[1][1].set_title('Vertical: \u03C3 = 1')
ax[2][0].set_title('Horizontal: \u03C3 = 2')
ax[2][1].set_title('Vertical: \u03C3 = 2')
ax[3][0].set_title('Horizontal: \u03C3 = 4')
ax[3][1].set_title('Vertical: \u03C3 = 4')
ax[4][0].set_title('Center Surround: $G_2(x,y)-G_1(x,y)$')
ax[4][1].set_title('Center Surround: $G_4(x,y)-G_2(x,y)$')

plt.show()
