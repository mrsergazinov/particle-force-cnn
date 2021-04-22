"""
Main function call.
"""

import sys
import numpy as np
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from photoelastic_response import *
from force_list import *

if __name__ == '__main__':
    # set particle parameters
    #num forces
    num_forces = 2     
    # number of different magnitudes for forces
    num_mags = 4
    # number of different angles 
    num_angles_inner = 4
    num_angles_tang = 4
    # number of randomly noised pictures for each force selection    
    num_random = 1
    
    # max/min magnitude for force
    lower_bound = 0.01
    upper_bound = 0.9
    
    # delta angle between contact forces
    delta_angle_inner = np.pi / 6

    #radius/height pf particle
    radius = [0.0055]
    height = 0.005
    
    # material constant
    f_sigma = 11000
    # pixels per radius
    px2m = 0.00019
    # brightness cutoff
    cutoff = np.infty
    
    # check input is supplied from command line
    if (sys.argv[1] == 'user_input'):
        radius = []
        num_radius = int(sys.argv[2])
        for i in range(num_radius):
            radius.append(float(sys.argv[3 + i]))
        num_forces = int(sys.argv[2 + num_radius + 1])
        num_mags = int(sys.argv[2 + num_radius + 2])
        num_angles_inner = int(sys.argv[2 + num_radius + 3])
        num_angles_tang = int(sys.argv[2 + num_radius + 4])
        num_random = int(sys.argv[2 + num_radius + 5])
        subset = sys.argv[2 + num_radius + 6]

    #multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #generate force lists
    list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, num_random, lower_bound, upper_bound, delta_angle_inner)
    #save force descriptions
    labels_angles_inner = np.array([[f.get_phi() for f in F] for F in list_of_F])
    labels_angles_tang = np.array([[f.get_alpha() for f in F] for F in list_of_F])
    labels_mags = np.array([[f.get_mag() for f in F] for F in list_of_F])
    np.save(os.path.join(os.getcwd(), 'labels', subset, str(num_forces), 'angles_inner.npy'),
                labels_angles_inner)
    np.save(os.path.join(os.getcwd(), 'labels', subset, str(num_forces), 'angles_tang.npy'),
            labels_angles_tang)
    np.save(os.path.join(os.getcwd(), 'labels', subset, str(num_forces), 'mags.npy'),
            labels_mags)
    
    for r in radius:
        #image generator with preset parameters
        particle = Particle(r, height)
        pixels_per_radius = int(round(r / px2m))
        image_gen_preset = partial(photo_elastic_response_on_particle, particle, f_sigma, pixels_per_radius, cutoff)
        
        #generate images of particles based on the force lists
        data = pool.map(image_gen_preset, list_of_F)
        
        for i in range(len(data)):
            plt.imsave(os.path.join(os.getcwd(), 'image_data', subset, str(num_forces), 
                                    'r' + str(r) + '_img' + str(i) + '.jpg'), np.asarray(data[i]), vmin = 0, vmax = 1, cmap = 'gray')

        print(r, num_radius, num_forces, num_mags, num_angles_inner, num_angles_tang, num_random)
    
