"""
This program reduces the color palette of an image into k distinct colors
It is able to separate these colors through k-means clustering
"""

import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans, vq

"""
Performs k-means clustering on a image into k distinct colors

Parameters
----------
image : numpy.array (dtype = float)
    an HxWX3 array of an image

clusters : int
    the number of distinct colors to represent the image
    
Returns
-------
numpy.array (dtype = float)
    an HXWX3 array of an image represented as k distinct colors
"""
def segment_image(image, clusters):
    
    #Reshapes image for use with k-means clustering
    h, w, d = image.shape
    image2d = image.reshape(h * w, d)
    
    #Obtain the centroids of image pixels calculated through k-means
    centroids, _ = kmeans(image2d, clusters)
    
    #Obtain the indices associated to the centroids based on image
    indices, _ = vq(image2d, centroids)
    indices = indices.reshape(h, w)
    
    #Obtain resulting image separated into k distinct colors
    result = centroids[indices]
    
    return result

"""
Scales the R value of an image by 1000
Performs k-means clustering on the scaled image into k distinct colors

Parameters
----------
image : numpy.array (dtype = float)
    an HxWX3 array of an image

clusters : int
    the number of distinct colors to represent the image
    
Returns
-------
numpy.array (dtype = float)
    an HXWX3 array of the scaled image represented as k distinct colors
"""
def segment_image_scaled(image, clusters):
    
    #Scale R value of image provided by a factor of 1000
    scaled = image.copy()
    scaled[:, :, 0] = scaled[:, :, 0] * 1000
    
    #Reshapes image for use with k-means clustering
    h, w, d = scaled.shape
    image2d = scaled.reshape(h * w, d)

    #Obtain the centroids of image pixels calculated through k-means
    centroids, _ = kmeans(image2d, clusters)
    
    #Obtain the indices associated to the centroids based on image
    indices, _ = vq(image2d, centroids)
    indices = indices.reshape(h, w)
    
    #Obtain resulting image separated into k distinct colors
    result = centroids[indices]
    
    #Scale R value of resulting image back to normal
    result[:, :, 0] = result[:, :, 0] / 1000
    
    return result

#Read colorful image
image = plt.imread('nature_resize.jpg')
image = image.astype('float')

#Obtain version of image separated into k distinct colors
k2 = segment_image(image, 2).astype('uint8')
k5 = segment_image(image, 5).astype('uint8')
k10 = segment_image(image, 10).astype('uint8')

#Obtain scaled version of image separated into k distinct colors
sk2 = segment_image_scaled(image, 2).astype('uint8')
sk5 = segment_image_scaled(image, 5).astype('uint8')
sk10 = segment_image_scaled(image, 10).astype('uint8')

image = image.astype('uint8')

#Plot results
fig, ax = plt.subplots(3, 3, figsize=(15, 12))

ax[0, 0].axis('off')
ax[0, 1].imshow(image)
ax[0, 1].set_title('Original Image')
ax[0, 2].axis('off')

ax[1, 0].imshow(k2)
ax[1, 0].set_title('k = 2')
ax[1, 1].imshow(k5)
ax[1, 1].set_title('k = 5')
ax[1, 2].imshow(k10)
ax[1, 2].set_title('k = 10')

ax[2, 0].imshow(sk2)
ax[2, 0].set_title('k = 2 (scaled)')
ax[2, 1].imshow(sk5)
ax[2, 1].set_title('k = 5 (scaled)')
ax[2, 2].imshow(sk10)
ax[2, 2].set_title('k = 10 (scaled)')

plt.show()
