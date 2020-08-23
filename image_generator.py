'''
This files defines the data generator for the images of particles.
'''
from photoelastic_response import *
import numpy as np

def image_gen(F):
    '''Function generates image based on list of forces'''
    #particle
    p = Particle(1, 0.1)
    # material constant
    actual_f_sigma = 1
    #resolution of the image will be n_pixels_per_radius*2-by-n_pixels_per_radius*2
    n_pixels_per_radius = 28
    # max possible value of stress
    Cutoff = 10 
    #noise coefficient
    coef_random = 0.1
    
    img = photo_elastic_response_on_particle(p, F, actual_f_sigma, n_pixels_per_radius, Cutoff)
    mu = np.max(img)
    sigma = np.std(img)
    for ix,iy in np.ndindex(img.shape):
        if (img[ix,iy] != 0): 
            img[ix,iy] += coef_random*np.random.normal(mu, sigma)
    return img

