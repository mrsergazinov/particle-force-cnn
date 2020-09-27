#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:30:54 2020

@author: renatsergazinov
"""

import matplotlib.pyplot as plt
import numpy as np
import os

image_path = os.path.join(os.getcwd(), "image_data")

X = np.load(os.path.join(image_path, "data.npy"))
num_images = X.shape[0]
predict_images = np.load(os.path.join(image_path, "predicted.npy"))
index = np.load(os.path.join(image_path, "index_img.npy"))

fig2 = plt.figure(figsize = (40, 20))
for i in range(2*num_images):
    fig2.add_subplot(2, num_images, i+1)
    plt.axis('off')
    if i < num_images:
        plt.imshow(np.asarray(X[index[i],:,:]), vmin= 0, vmax = 1, cmap='gray')
    else:
        plt.imshow(np.asarray(predict_images[i-num_images,:,:]), vmin= 0, vmax = 1, cmap='gray')