"""
Main function call.
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from image_generator import *
from force_list import *

if __name__ == '__main__':
    #max/min magnitude for force
    lower_bound = 0.2
    upper_bound = 0.4
    #number of randomly noised pictures for each force selection
    num_random = 1
    #max/min number of forces
    max_num_forces = 6
    min_num_forces = 2
    #number of different magnitudes for forces
    num_mags = 3
    #number of different angles 
    num_angles_inner = 2
    num_angles_tang = 2
    #num pixles in image *2
    n_pixels_per_radius = 28
    #num forces
    num_forces = 5
    
    #multiprocessing
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, num_random, lower_bound, upper_bound)
    labels_num_forces = np.zeros(len(list_of_F)) + num_forces
    labels_angles_inner = np.array([[f.get_phi() for f in F] for F in list_of_F])
    labels_angles_tang = np.array([[f.get_alpha() for f in F] for F in list_of_F])
    labels_mags = np.array([[f.get_mag() for f in F] for F in list_of_F])
    data = np.array(pool.map(image_gen, list_of_F))
    

    fig = plt.figure(figsize = (25, 25))
    for i in range(12):
        fig.add_subplot(4, 3, i+1)
        plt.imshow(np.asarray(data[i,:,:]), vmin= 0, vmax = 1, cmap='gray')
        
# =============================================================================
#np.save('la2', labels_angles)
#np.save('lm2', labels_mags)
#np.save('lnf2', labels_num_forces)
#np.save('data2', data)
# =============================================================================
        
    
