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
    #number of different magnitudes for forces
    num_mags = 4
    #number of different angles 
    num_angles_inner = 4
    num_angles_tang = 4
    #number of randomly noised pictures for each force selection    
    num_random = 1
    
    #max/min magnitude for force
    lower_bound = 0.0001
    upper_bound = 0.4

    #radius/height pf particle
    radius = 0.0055
    height = 1
    
    # material constant
    f_sigma = 100.24
    # pixels per radius
    pixels_per_radius = 40
    # brightness cutoff
    cutoff = 10000
    
# check if input is supplied from command line
    if (len(sys.argv) == 2):
        num_forces = int(sys.argv[1])
    elif (len(sys.argv) == 6):
        num_forces = int(sys.argv[1])
        num_mags = int(sys.argv[2])
        num_angles_inner = int(sys.argv[3])
        num_angles_tang = int(sys.argv[4])
        num_random = int(sys.argv[5])
    elif (len(sys.argv) == 13):
        num_forces = int(sys.argv[1])
        num_mags = int(sys.argv[2])
        num_angles_inner = int(sys.argv[3])
        num_angles_tang = int(sys.argv[4])
        num_random = int(sys.argv[5])
        lower_bound = float(sys.argv[6])
        upper_bound = float(sys.argv[7])
        radius = float(sys.argv[8])
        height = float(sys.argv[9])
        f_sigma = float(sys.argv[10])
        pixels_per_radius = int(sys.argv[11])
        cutoff = float(sys.argv[12])
    
    #multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #image generator with preset parameters
    particle = Particle(radius, height)
    image_gen_preset = partial(photo_elastic_response_on_particle, particle, f_sigma, pixels_per_radius, cutoff)
    
    #generate force lists
    list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, num_random, lower_bound, upper_bound)
    #save force descriptions for each particle
    labels_angles_inner = np.array([[f.get_phi() for f in F] for F in list_of_F])
    labels_angles_tang = np.array([[f.get_alpha() for f in F] for F in list_of_F])
    labels_mags = np.array([[f.get_mag() for f in F] for F in list_of_F])
    #generate images of particles based on the force lists
    data = pool.map(image_gen_preset, list_of_F)
    
    print(num_forces)
    # save images
    for i in range(len(data)):
        plt.imsave(os.path.join(os.getcwd(), 'image_data', 'train', str(num_forces), 'img' + str(i) + '.jpg'),
                   np.asarray(data[i]), 
                   vmin= 0, vmax = 1, 
                   cmap='gray')
    np.save(os.path.join(os.getcwd(), 'labels', 'train', str(num_forces), 'angles_inner.npy'),
            labels_angles_inner)
    np.save(os.path.join(os.getcwd(), 'labels', 'train', str(num_forces), 'angles_tang.npy'),
            labels_angles_tang)
    np.save(os.path.join(os.getcwd(), 'labels', 'train', str(num_forces), 'mags.npy'),
            labels_mags)

        
    
