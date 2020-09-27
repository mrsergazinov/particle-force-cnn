#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs test functions.
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from test_functions import *


if __name__ == '__main__':
    min_num_forces = 2
    max_num_forces = 6
    
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    models_path = os.path.join(os.getcwd(), 'saved_models')
    image_path = os.path.join(os.getcwd(), "image_data", "data.npy")
    
    X = load_data(image_path)
    num_images = X.shape[0]
    
    (model_class, models_ai, models_at, models_m) = load_models(models_path, min_num_forces,
                                                                max_num_forces)
    (predict_ai, predict_m, predict_at, num_forces, index) = predict(X, model_class, models_ai, 
                                                  models_at, models_m)
    min_num_forces, max_num_forces = min(num_forces), max(num_forces)
    F = generate_F_lists(predict_ai, predict_at, predict_m, max_num_forces)
    
    predict_images = np.array(pool.map(image_gen, F))
    X = X.squeeze(axis = 3)
    np.save(os.path.join(os.getcwd(), "image_data", "predicted.npy"), predict_images)
    np.save(os.path.join(os.getcwd(), "image_data", "index_img.npy"), index)
    # fig2 = plt.figure(figsize = (40, 20))
    # for i in range(2*num_images):
    #     fig2.add_subplot(2, num_images, i+1)
    #     plt.axis('off')
    #     if i < num_images:
    #         plt.imshow(np.asarray(X[index[i],:,:]), vmin= 0, vmax = 1, cmap='gray')
    #     else:
    #         plt.imshow(np.asarray(predict_images[i-num_images,:,:]), vmin= 0, vmax = 1, cmap='gray')
        



