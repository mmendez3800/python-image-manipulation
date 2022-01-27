"""
This program obtains the phase and magnitude of two images
From there, it produces two new images
One is the phase of the first image with the magnitude of the second
The other is the phase of the second image with the magnitude of the first
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from numpy import fft
from PIL import Image

#Open images and conver to numpy arrays
img1 = Image.open(os.getcwd() + '/giraffe_resize.jpg').convert('L')
img1 = np.array(img1, dtype = 'float') / 256
img2 = Image.open(os.getcwd() + '/tiger_resize.jpg').convert('L')
img2 = np.array(img2, dtype = 'float') / 256

#Obtain DFTs of images
dft1 = fft.fft2(img1)
dft1 = fft.fftshift(dft1)
dft2 = fft.fft2(img2)
dft2 = fft.fftshift(dft2)

#Calculate magnitude spectrums and phase spectrums of DFTs
mag1 = np.abs(dft1)
mag2 = np.abs(dft2)
phase1 = np.exp(1j * np.angle(dft1))
phase2 = np.exp(1j * np.angle(dft2))

#Combine the opposite spectrums of the images to create a new image
comb1 = np.multiply(mag2, phase1)
comb2 = np.multiply(mag1, phase2)

#Revert combined spectrums back into 2D images
comb1 = fft.ifftshift(comb1)
comb1 = fft.ifft2(comb1)
comb2 = fft.ifftshift(comb2)
comb2 = fft.ifft2(comb2)

#Display results
fig, axs = plt.subplots(2, 2, figsize = (10, 10))
axs[0][0].imshow(img1, cmap = 'gray')
axs[0][0].set_title("Image 1")
axs[0][1].imshow(img2, cmap = 'gray')
axs[0][1].set_title("Image 2")
axs[1][0].imshow(np.real(comb1), cmap = 'gray')
axs[1][0].set_title("Phase = Image 1 & Magnitude = Image 2")
axs[1][1].imshow(np.real(comb2), cmap = 'gray')
axs[1][1].set_title("Phase = Image 2 & Magnitude = Image 1")
plt.show()
