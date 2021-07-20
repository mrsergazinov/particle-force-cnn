#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code defines functions to test CNN models on images.
"""

import tensorflow as tf
""" physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True) """

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from classes import *

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
    num_forces = model_class.predict(X).argmax(-1)+2
    predict_ai, predict_at, predict_m = list(), list(), list()
    
    for i in num_forces:
        predict_ai[i] = models_ai[num_forces[i]].predict(X[None, i, ])[1].tolist()
        predict_at[i] = models_at[num_forces[i]].predict(X[None, i, ])[1].tolist()
        predict_m[i] = models_m[num_forces[i]].predict(X[None, i, ])[1].tolist()
        
    return (lai_pred, lm_pred, lat_pred, num_forces)

def generate_F_lists(predict_ai, predict_at, predict_m, num_forces): 
    ''' The function outputs the description of the predicted list of forces for each particle. '''
    list_of_F_pred = []
    for i in range(0, len(num_forces)):
        F_pred = [] 
        for j in range(0, num_forces[i]):
            F_pred.append(Force(predict_m[i][j], predict_ai[i][j], predict_at[i][j]))
        list_of_F_pred.append(F_pred)
    return list_of_F_pred
    

