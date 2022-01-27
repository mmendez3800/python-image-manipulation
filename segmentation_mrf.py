"""
This program builds a Markov Random Field model
From there, the user can interactively segment an image
It will allow them to see the foreground and background
"""

import matplotlib.pyplot as plt
import numpy as np
import maxflow

from scipy.spatial import distance
from selectpoints import select_k_points

"""
Create a segmentation of a given image into the foreground and background
components of the image

Parameters
----------
image : numpy.array(dtype = int)
    the image to be used for segmentation

foreground : list
    the coordinates of the section of the image indicated to be a
    part of the foreground

background : list
    the coordinates of the section of the image indicated to be a
    part of the background
    
lmda : float
    the lambda value that is multiplied to the edges associaed to neighboring
    pixels to indicate their relative importance in the graph
    
Returns
-------
numpy.array (dtype = int)
    an HXW array containing the results of segmenting the image into the
    foreground and background components
"""

def foreback_segment(image, foreground, background, lmda):
        
    """
    Given an image and a set of coordinates, determines the average RGB channels
    of the pixel based on the neighbors of the pixel

    Parameters
    ----------
    image : numpy.array(dtype = int)
        the image to be used for obtaining RGB channel averages

    ground : list
        the coordinates of the image to obtain the calculated avergaes

    Returns
    -------
    numpy.array (dtype = float)
        an array containing the average RGB channel values of the pixel and its eight
        surrounding neighbors
    """
    def get_averages(img, ground):
        
        red = img[(ground[0] - 1):(ground[0] + 2), (ground[1] - 1):(ground[1] + 2), 0]
        green = img[(ground[0] - 1):(ground[0] + 2), (ground[1] - 1):(ground[1] + 2), 1]
        blue = img[(ground[0] - 1):(ground[0] + 2), (ground[1] - 1):(ground[1] + 2), 2]
        
        result = np.array([np.mean(red), np.mean(green), np.mean(blue)], dtype='float')
        
        return result
    
    copy = np.copy(image)
    copy = copy.astype('float')
    
    #Obtain average RGB values of image based on foreground and background coordinates
    fore = get_averages(copy, foreground)
    back = get_averages(copy, background)
    
    h, w, _ = copy.shape
    
    #Create graph and create nodeids for each pixel in the image
    g = maxflow.GraphFloat()
    nodeids = g.add_grid_nodes((h, w))
    
    for i in range(h):
        for j in range(w):
            
            #Calculate distance from each point in image to average
            slink = distance.euclidean(copy[i, j, :], fore)
            tlink = distance.euclidean(copy[i, j, :], back)
            
            #Add st edges for each nodeid
            g.add_tedge(nodeids[i][j], slink, tlink)
            
            #If pixel is not edge pixel, create horizontal links between neighboring pixels
            if (j + 1) < w:
                dist = distance.euclidean(copy[i, j, :], copy[i, (j + 1), :]) * lmda
                g.add_edge(nodeids[i][j], nodeids[i][j + 1], dist, dist)
            
            #If pixel is not edge pixel, create vertilcal links between neighboring pixels
            if (i + 1) < h:
                dist = distance.euclidean(copy[i, j, :], copy[(i + 1), j, :]) * lmda
                g.add_edge(nodeids[i][j], nodeids[i + 1][j], dist, dist)
      
    #Perform mincut on graph and obtain results of which pixels are linked to which component
    g.maxflow()
    segments = g.get_grid_segments(nodeids)
    results = np.int_(np.logical_not(segments))
    
    return results
#Read image and display for user to select points
image = plt.imread('starfish_resize.jpg')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image)
spoints = select_k_points(ax, 2)

#Locate selected points
print('Performing Segmentation')
xpoints = spoints.xs
ypoints = spoints.ys
foreground = [round(ypoints[0].item(0)), round(xpoints[0].item(0))]
background = [round(ypoints[1].item(0)), round(xpoints[1].item(0))]

#Obtain results of foreground/background selection
result = foreback_segment(image, foreground, background, 2)

#Display results
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.plot(foreground[1], foreground[0], 'rx')
ax1.plot(background[1], background[0], 'rx')
ax2.imshow(result, cmap='gray')
ax2.set_title('Segmented Image')
plt.show()
