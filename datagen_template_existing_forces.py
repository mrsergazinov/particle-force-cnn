"""
This file contains functions for computing photoelastic response of a disk particle
positioned  on the (X,Y) plane with it center at the origin.

We use two types of coordinates:
 1) standard cartesian coordinates on the plane (X,Y) to which we refer as xy coordinates
 2) Polar coordinates based at the force impact. Zero angle pointing at the direction
    of the force and angle increasing counter clockwise to which we refer as f_polar coordinates

 The photoelastic intensity field is in XY coordinates
"""

import numpy as np
from math import sin, cos, asin, acos, pi, sqrt
from multiprocessing import Pool, cpu_count
import os
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, radius=1, height=1):
        self.radius = radius
        self.height = height

"""
Each force acting on this particle at its boundary is described by three variables 
"""

class Force:
    def __init__(self, magnitude=0, phi=0, alpha=0):
        """ force magnitude """
        self.magnitude = magnitude
        """ Angle between the origin of XY (center of the particle) and the point of impact
            of the force (on the boundary of the particle ) measured counter clockwise from 
            the x-axis
        """
        self.phi = phi
        """ Impact angle of the force is the angle between the normal of the circle at the force
            impact point pointing inward and the direction of the force.
            It is measured counter clockwise form the inward pointing normal.  
        """
        self.alpha = alpha

    # Angle between the force vector and the X coordinate measured counter clockwise
    @property
    def direction_angle_xy(self):
        return self.phi + self.alpha + np.pi

    # Unit vector in the force direction in xy coordinates
    @property
    def unit_force_vector_xy(self):
        angle_from_x = self.direction_angle_xy
        return np.cos(angle_from_x), np.sin(angle_from_x)
    
    def get_mag(self):
        return self.magnitude
    def get_phi(self):
        return self.phi
    def get_alpha(self):
        return self.alpha

def force_impact_point_xy(particle, force):
    """ 
        Returns the xy coordinates of the point at which the force acts 
        on the particle
    """
    x = particle.radius * np.cos(force.phi) 
    y = particle.radius * np.sin(force.phi)
    return x, y

def xy_to_f_polar(x, y, particle, force):
    impact_x, impact_y = force_impact_point_xy(particle, force)
    unit_force_x, unit_force_y = force.unit_force_vector_xy
    # compute vector between the impact point and the point (x,y) in xy coordinates
    xx = x - impact_x
    yy = y - impact_y
    
    # r part of the f_polar coordinates 
    f_polar_r = np.sqrt(np.square(xx) + np.square(yy))

    cos_theta = (xx * unit_force_x + yy * unit_force_y) / f_polar_r
    # make sure that cos_theta is in the right range (rounding can compromise this)
    if cos_theta > 1 :
        cos_theta = 1
    if cos_theta < -1 :
        cos_theta = -1

    # angle part of the f_polar coordinates 
    f_polar_theta = np.arccos(cos_theta)
    if unit_force_x * yy - unit_force_y * xx < 0:
        f_polar_theta = 2 * np.pi - f_polar_theta

    return f_polar_r, f_polar_theta

def stress_tensor_rr_in_f_polar(f_polar_r, f_polar_theta, particle, force, cutoff=np.inf):
    """
        Returns the rr part of the stress tensor in f_polar coordinates
        All the other parts of the tensor are zero in this coordinates
        To make it more realistic cutoff can be set to maximal possible value of the stress
    """
    c = force.magnitude / (np.pi * particle.height)
    force_stress = 2*np.cos(f_polar_theta) / f_polar_r
    boundary_balance_stress = -np.cos(force.alpha) / particle.radius
    stress_tensor_rr = c*(force_stress + boundary_balance_stress)
    return min(stress_tensor_rr, cutoff)

def transform_stress_tensor_in_f_polar_to_xy(stress_tensor_rr, f_polar_theta, force):
    '''Function computes force in xy-coordinates given polar coordinates'''
    sin_rotation = np.sin(f_polar_theta + force.phi + force.alpha)
    cos_rotation = np.cos(f_polar_theta + force.phi + force.alpha)
    stress_tensor_xx = stress_tensor_rr * np.square(cos_rotation)
    stress_tensor_yy = stress_tensor_rr * np.square(sin_rotation)
    stress_tensor_xy = stress_tensor_rr * sin_rotation * cos_rotation
    return np.array([stress_tensor_xx, stress_tensor_xy, stress_tensor_yy])

def stress_tensor_in_xy_one_force(x, y, particle, force, cutoff=np.inf):
    '''Function computes stress tensor given one force'''
    r, theta = xy_to_f_polar(x, y, particle, force)
    sigma_rr = stress_tensor_rr_in_f_polar(r, theta, particle, force, cutoff)
    return transform_stress_tensor_in_f_polar_to_xy(sigma_rr, theta, force)

def stress_tensor_in_xy(x, y, particle, forces, cutoff=np.inf):
    '''Function cpmutes stress tensors given forces'''
    sigma_xx = 0
    sigma_xy = 0
    sigma_yy = 0
    for force in forces:
        sigma_xx_f, sigma_xy_f, sigma_yy_f = stress_tensor_in_xy_one_force(x, y, particle, force, cutoff)
        sigma_xx += sigma_xx_f
        sigma_xy += sigma_xy_f
        sigma_yy += sigma_yy_f

    return sigma_xx, sigma_xy, sigma_yy

def photo_elastic_response_from_stress(stress_tensor_xx, stress_tensor_xy, stress_tensor_yy, f_sigma):
    '''Function computes photoelastic reponse at apoint given stress tensors'''
    principal_stress_diff = np.sqrt(np.square(stress_tensor_xx - stress_tensor_yy) + 4 * np.square(stress_tensor_xy))
    return np.square(np.sin(np.pi * principal_stress_diff / f_sigma))

def photo_elastic_response_at_xy(x, y, particle, forces, f_sigma, cutoff=np.inf):
    '''Function computes photoelastic reponse at apoint given forces'''
    sigma_xx, sigma_xy, sigma_yy = stress_tensor_in_xy(x, y, particle, forces, cutoff)
    return photo_elastic_response_from_stress(sigma_xx, sigma_xy, sigma_yy, f_sigma)


def photo_elastic_response_on_particle(particle, forces, f_sigma, pixels_per_radius, cutoff=np.inf):
    '''Function computes photoelastic reponse at each point'''
    photo_elastic_response = np.zeros((2 * pixels_per_radius, 2 * pixels_per_radius))
    pixel_to_coordinate = np.linspace(-particle.radius, particle.radius, 2 * pixels_per_radius)
    radius_sqr = np.square(particle.radius)
    
    for i in np.arange(2 * pixels_per_radius):
        for j in np.arange(2 * pixels_per_radius):
            if np.square(pixel_to_coordinate[i]) + np.square(pixel_to_coordinate[j]) < radius_sqr:
                photo_elastic_response[i, j] = photo_elastic_response_at_xy(pixel_to_coordinate[i], pixel_to_coordinate[j], particle, forces, f_sigma, cutoff)
            else:
                photo_elastic_response[i, j] = 0
    return photo_elastic_response

def test(particle, force, f_sigma, pixels_per_radius, cutoff=np.inf):
    t = np.zeros((2 * pixels_per_radius, 2 * pixels_per_radius))
    pixel_to_coordinate = np.linspace(-particle.radius, particle.radius, 2 * pixels_per_radius)
    radius_sqr = np.square(particle.radius)
    for i in np.arange(2 * pixels_per_radius):
        for j in np.arange(2 * pixels_per_radius):
            if np.square(pixel_to_coordinate[i]) + np.square(pixel_to_coordinate[j]) < radius_sqr:
                r, p = xy_to_f_polar(pixel_to_coordinate[i], pixel_to_coordinate[j], particle, force)
                st = stress_tensor_rr_in_f_polar(r, p, particle, force, cutoff)
                # t[i, j] = r
                # t[i, j] = phi
                #t[i, j] = stress_tensor_rr_in_f_polar(r, p, particle, force)
                xx,xy,yy =transform_stress_tensor_in_f_polar_to_xy(st, p, force)
                t[i,j] = st
            else:
                t[i, j] = 0
    return t

def image_gen(F):
    '''Function generates image based on list of forces'''
    # particle
    p = Particle(1, 0.1)
    # material constant
    actual_f_sigma = 1
    # resolution of the image will be n_pixels_per_radius*2-by-n_pixels_per_radius*2
    n_pixels_per_radius = 28
    # max possible value of stress
    Cutoff = 10 
    #noise coefficient
    coef_random = 0.1
    
    img = photo_elastic_response_on_particle(p, F, actual_f_sigma, n_pixels_per_radius, Cutoff)
    # mu = np.max(img)
    # sigma = np.std(img)
    # for ix,iy in np.ndindex(img.shape):
    #     if (img[ix,iy] != 0): 
    #         img[ix,iy] += coef_random*np.random.normal(mu, sigma)
    return img

if __name__ == '__main__':
    #num pixles in image *2
    n_pixels_per_radius = 28
    
    #multiprocessing with 4 processes
    num_processes = cpu_count()
    pool = Pool(processes = num_processes)
    
    minn = 2
    maxx = 6
    path = os.path.join(os.getcwd(), 'data')
    
    lai_pred, lm_pred, lat_pred = np.load(os.path.join(path, 'lai_pred2.npy')), np.load(os.path.join(path, 'lm_pred2.npy')), np.load(os.path.join(path, 'lat_pred2.npy'))
    lai_true, lm_true, lat_true = np.load(os.path.join(path, 'lai_true2.npy')), np.load(os.path.join(path, 'lm_true2.npy')), np.load(os.path.join(path, 'lat_true2.npy'))
    
    for i in range(minn+1, maxx+1):
        lai_pred = np.vstack([lai_pred, np.load(os.path.join(path, 'lai_pred'+str(i)+'.npy'))])
        lat_pred = np.vstack([lat_pred, np.load(os.path.join(path, 'lat_pred'+str(i)+'.npy'))])
        lm_pred = np.vstack([lm_pred, np.load(os.path.join(path, 'lm_pred'+str(i)+'.npy'))])                      
        lai_true = np.vstack([lai_true, np.load(os.path.join(path, 'lai_true'+str(i)+'.npy'))]) 
        lat_true = np.vstack([lat_true, np.load(os.path.join(path, 'lat_true'+str(i)+'.npy'))]) 
        lm_true = np.vstack([lm_true, np.load(os.path.join(path, 'lm_true'+str(i)+'.npy'))])
    
    list_of_F = []
    for l in range(lai_pred.shape[0]):
        F_true = []
        for i in range(maxx):
            F_true.append(Force(lm_true[l, i], lai_true[l,i], lat_true[l, i]))
            
        list_of_F.append(F_true)
        
    for l in range(lai_pred.shape[0]):
        F_pred = [] 
        for i in range(maxx):
            F_pred.append(Force(lm_pred[l, i], lai_pred[l,i], lat_pred[l, i]))
        list_of_F.append(F_pred)
    
    data = np.array(pool.map(image_gen, list_of_F))
    fig = plt.figure(figsize = (50, 10))
    for i in range(18):
        fig.add_subplot(2, 9, i+1)
        plt.axis('off')
        plt.imshow(np.asarray(data[i,:,:]), vmin= 0, vmax = 1, cmap='gray')
        
        
        
    
