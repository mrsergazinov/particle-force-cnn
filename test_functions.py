#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code defines functions to test CNN models on images.
"""

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from datagen_template import *
from classes import *

def load_data(path):
    '''The function loads images from the given path. Images should be greyscale in form of 
    Numpy matrix, where each entry corresponds to pixel value. The entries of the image 
    matrix should be scaled to the interval [0,1].
    
    The image matrices should be sotred as a tensor, where [i,j,k]^th element correponds 
    to the (j,k)^th pixel value in the i^th image.'''
    
    X = np.load(path)
    
    #make images appear as having 1 color layer, if they are not already strcutred so
    if (len(X.shape) != 4):
        if K.image_data_format() == 'channels_first':
            X = np.repeat(X[np.newaxis, ...], 1, axis = 0)
        else:
            X = np.repeat(X[..., np.newaxis], 1, axis = -1)
    
    return X

def load_models(path, min_num_forces, max_num_forces):
    ''' Load CNN models from the given path. Min_num_forces and max_num_forces
    lets the user select the minimum and the maximum number of forces that she/he believes 
    could be acting on the particle. The CNN are loaded using Tensorflow - Keras.
    
    Currently, we have models for  {2,3,4,5,6} forces.'''
    model_class = load_model(os.path.join(path, 'class_num_forces.h5'))
                            
    models_ai = dict()
    models_at = dict()
    models_m = dict()                         
    for i in range(min_num_forces, max_num_forces+1):
        models_ai[i] = load_model(os.path.join(path, 'reg_num_angles_'+str(i)+'.h5'))
        models_at[i] = load_model(os.path.join(path, 'reg_num_angles_tang_'+str(i)+'.h5'))
        models_m[i] = load_model(os.path.join(path, 'reg_num_mags_'+str(i)+'.h5'))
    return (model_class, models_ai, models_at, models_m)

def predict(X, model_class, models_ai, models_at, models_m):
    ''' The function predicts the force list based on the image of the particle. It then 
    outputs, the predicted position angles, tangent angles, and magnitudes as Numpy vectors.
    
    The outputs are padded with zeros to match the max_num_forces.'''
    predict = model_class.predict(X).argmax(-1)+2
    num_forces = list(set(predict))
    max_num_forces = max(num_forces)
    
    X_class_pred = dict()
    index = []
    predict_ai, predict_at, predict_m = dict(), dict(), dict()
    
    for i in num_forces:
        X_class_pred[i] = X[np.argwhere(predict == i).flatten().tolist()]
        index += np.argwhere(predict == i).flatten().tolist()
    for i in num_forces:
        predict_ai[i] = models_ai[i].predict(X_class_pred[i])
        predict_at[i] = models_at[i].predict(X_class_pred[i])
        predict_m[i] = models_m[i].predict(X_class_pred[i])
        predict_ai[i] = np.hstack([predict_ai[i], np.zeros([predict_ai[i].shape[0], max_num_forces - predict_ai[i].shape[1]])])
        predict_at[i] = np.hstack([predict_at[i], np.zeros([predict_at[i].shape[0], max_num_forces - predict_at[i].shape[1]])])
        predict_m[i] = np.hstack([predict_m[i], np.zeros([predict_m[i].shape[0], max_num_forces - predict_m[i].shape[1]])])
    
    lai_pred, lm_pred, lat_pred = predict_ai[num_forces[0]], predict_m[num_forces[0]], predict_at[num_forces[0]]
    for i in num_forces[1:]:
        lai_pred = np.vstack((lai_pred, predict_ai[i]))
        lat_pred = np.vstack((lat_pred, predict_at[i]))
        lm_pred = np.vstack((lm_pred, predict_m[i]))
        
    return (lai_pred, lm_pred, lat_pred, num_forces, index)

def generate_F_lists(predict_ai, predict_at, predict_m, max_num_forces): 
    ''' The function generates particle image matrices based on the list position angles,
    tangent angles, and magnitudes. It outputs the matrices as Numpy matrix.'''

    list_of_F_pred = []
    for l in range(predict_ai.shape[0]):
        F_pred = [] 
        for i in range(max_num_forces):
            F_pred.append(Force(predict_m[l, i], predict_ai[l,i], predict_at[l, i]))
        list_of_F_pred.append(F_pred)
    
    return list_of_F_pred
    

