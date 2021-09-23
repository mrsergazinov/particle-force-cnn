#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code runs test functions.
"""
# external libs
import tensorflow as tf
from tensorflow.keras.models import load_model
import h5py
import sys
import pickle
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import os
# internal libs
from tools.prediction import *

if __name__ == '__main__':
    #particle parameters to set
    r = 0.008
    height = 0.0055
    particle = Particle(r, height)
    f_sigma = 11000
    px2m = 0.00019
    pixels_per_radius = int(round(r / px2m))
    cutoff = np.infty

    #number of forces suspected present in the images: between 2 and 6
    min_num_forces = 2
    max_num_forces = 6

    #path to images: either user or pre-specified
    images_path_prefix = os.path.join(os.getcwd(), "image_data")
    #path to models
    models_path = os.path.join(os.getcwd(), 'models')

    # check input is supplied from command line
    if (sys.argv[1] == 'user_input'):
        r = int(sys.argv[2])
        min_num_forces = int(sys.argv[3])
        max_num_forces = int(sys.argv[4])
        images_path_prefix = sys.argv[5]
        models_path = sys.argv[6]
    
    #initializing multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #loading data
    images_names = sorted(os.listdir(images_path_prefix))
    images_path = [os.path.join(images_path_prefix, name) for 
                   name in images_names]
    #apply data generators
    params = {'dim': (128, 128), 
              'n_channels': 3, 
              'rescale': 1 / 255}
    data_generator = DataGenerator(images_path, **params)
    #get data
    X = data_generator.generate()
    
    #loading models
    (model_class, models_ai, models_at, models_m) = load_models(models_path, min_num_forces,
                                                                max_num_forces)
    
    #predict contact angles, tangent angles, magnitudes, number of forces for each particle
    (predict_ai, predict_at, predict_m, num_forces) = predict(X, model_class, models_ai, models_at, models_m)
    F = generate_F_lists(predict_ai, predict_at, predict_m, num_forces)
    
    # save predictions
    with open(os.path.join(images_path_prefix, "position_angles.txt"), "wb") as fp:
        pickle.dump(predict_ai, fp)
    with open(os.path.join(images_path_prefix, "tangent_angles.txt"), "wb") as fp:
        pickle.dump(predict_at, fp)
    with open(os.path.join(images_path_prefix, "magnitudes.txt"), "wb") as fp:
        pickle.dump(predict_m, fp)  
    
    #save predicted images
    image_gen_preset = partial(photo_elastic_response_on_particle, particle, 
                                f_sigma, pixels_per_radius, cutoff)
    predict_images = np.array(pool.map(image_gen_preset, F))
    for i in range(len(predict_images)):
        plt.imsave(os.path.join(images_path_prefix, 'predicted_' + images_names[i]),
                    np.asarray(predict_images[i]), vmin = 0, vmax = 1, cmap = 'gray')
    
        



