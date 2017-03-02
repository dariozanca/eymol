'''
Created on 31 ago 1948
@author: Dario Zanca 
@summary: Sample code to generate saliency map with EYMOL
'''

import eymol

import cv2
import numpy as np
import matplotlib.pyplot as plt

# upload an image
img = cv2.imread('051.jpg',0)

# compute the saliency map
sm = eymol.sm(img)

# plot results
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img, cmap='gray')
fig.add_subplot(1,2,2)
plt.imshow(sm, cmap='gray')
plt.show()

print "\nTHE END!\n"




