#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code generates data.
"""
# external libs
import sys
import numpy as np
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
# internal libs
from tools.photoelastic import *
from tools.force import *

if __name__ == '__main__':
    # set particle parameters
    num_forces = 2  #num forces
    num_mags = 2 # number of different magnitudes for forces
    num_angles_inner = 2 # number of different angles
    num_angles_tang = 2
    lower_bound = 0.01 # max/min magnitude for force
    upper_bound = 0.9
    delta_angle_inner = np.pi / 6  # delta angle between contact forces
    radius = 0.008 # radius/height pf particle
    height = 0.0055
    f_sigma = 11000 # material constant
    px2m = 0.00019 # pixels per radius
    cutoff = np.infty # brightness cutoff

    # set directories to save to
    path_label = os.path.join(os.getcwd(), 'labels')
    path_images = os.path.join(os.getcwd(), 'image_data')
    
    # check input is supplied from command line
    if (sys.argv[1] == 'user_input'):
        radius = int(sys.argv[2])
        num_forces = int(sys.argv[3])
        num_mags = int(sys.argv[4])
        num_angles_inner = int(sys.argv[5])
        num_angles_tang = int(sys.argv[6])
        path_label = sys.argv[7]
        path_images = sys.argv[8]

    #multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #generate force lists
    list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, lower_bound, upper_bound, delta_angle_inner)
    
    #save force descriptions
    labels_angles_inner = np.array([[f.get_phi() for f in F] for F in list_of_F])
    labels_angles_tang = np.array([[f.get_alpha() for f in F] for F in list_of_F])
    labels_mags = np.array([[f.get_mag() for f in F] for F in list_of_F])
    np.save(os.path.join(path_label, 'angles_inner.npy'),
                labels_angles_inner)
    np.save(os.path.join(path_label, 'angles_tang.npy'),
            labels_angles_tang)
    np.save(os.path.join(path_label, 'mags.npy'),
            labels_mags)
    
    # generate images
    #image generator with preset parameters
    particle = Particle(radius, height)
    pixels_per_radius = int(round(radius / px2m))
    image_gen_preset = partial(photo_elastic_response_on_particle, particle, f_sigma, pixels_per_radius, cutoff)

    #generate images of particles based on the force lists
    data = pool.map(image_gen_preset, list_of_F)

    for i in range(len(data)):
        plt.imsave(os.path.join(path_images, 'r' + str(radius) + '_img' + str(i) + '.jpg'), 
            np.asarray(data[i]), vmin = 0, vmax = 1, cmap = 'gray')

    print(radius, num_radius, num_forces, num_mags, num_angles_inner, num_angles_tang)
    
