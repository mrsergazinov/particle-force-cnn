#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs test functions.
"""

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os

from test_functions import *


if __name__ == '__main__':
    min_num_forces = 2
    max_num_forces = 6
    
    #initializing multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #defining paths
    models_path = os.path.join(os.getcwd(), 'saved_models')
    image_path = os.path.join(os.getcwd(), "image_data", "data.npy")
    
    #loading data
    X = load_data(image_path)
    num_images = X.shape[0]
    
    #loading models
    (model_class, models_ai, models_at, models_m) = load_models(models_path, min_num_forces,
                                                                max_num_forces)
    #making predictions
    (predict_ai, predict_m, predict_at, num_forces, index) = predict(X, model_class, models_ai, 
                                                  models_at, models_m)
    min_num_forces, max_num_forces = min(num_forces), max(num_forces)
    #generating force lists for each particle based on the predictions
    F = generate_F_lists(predict_ai, predict_at, predict_m, max_num_forces)
    #generating images of particles based on the predcited force lists
    image_gen_preset = partial(photo_elastic_response_on_particle, Particle(1, 0.1), 1, 28, 10)
    predict_images = np.array(pool.map(image_gen_preset, F))
    X = X.squeeze(axis = 3)
    #saving predicted paritcles' images
    np.save(os.path.join(os.getcwd(), "image_data", "predicted.npy"), predict_images)
    np.save(os.path.join(os.getcwd(), "image_data", "index_img.npy"), index)
        



