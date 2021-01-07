"""
Main function call.
"""

import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from photoelastic_response import *
from force_list import *

if __name__ == '__main__':
    #max/min magnitude for force
    lower_bound = 0.0001
    upper_bound = 0.02
    #number of randomly noised pictures for each force selection
    num_random = 1
    #number of different magnitudes for forces
    num_mags = 3
    #number of different angles 
    num_angles_inner = 2
    num_angles_tang = 2
    #num forces
    num_forces = 3
   
    #multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    #image generator with preset parameters
    particle = Particle(0.0154, 0.0031)
    f_sigma = 100.24
    pixels_per_radius = 40
    cutoff = 10000
    image_gen_preset = partial(photo_elastic_response_on_particle, particle, f_sigma, pixels_per_radius, cutoff)
    
    #generate force lists
    list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, num_random, lower_bound, upper_bound)
    #save force descriptions for each particle
    labels_num_forces = np.zeros(len(list_of_F)) + num_forces
    labels_angles_inner = np.array([[f.get_phi() for f in F] for F in list_of_F])
    labels_angles_tang = np.array([[f.get_alpha() for f in F] for F in list_of_F])
    labels_mags = np.array([[f.get_mag() for f in F] for F in list_of_F])
    #generate images of particles based on the force lists
    data = np.array(pool.map(image_gen_preset, list_of_F))
    
    #plot images
    fig = plt.figure(figsize = (25, 25))
    for i in range(12):
        fig.add_subplot(8, 3, i+1)
        plt.imshow(np.asarray(data[i,:,:]), vmin= 0, vmax = 1, cmap='gray')

        
    
