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
import pandas as pd
import os
from math import sin, cos, asin, acos, pi, sqrt
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

class Particle:
    def __init__(self, x, y, radius=1, height=1):
        self.radius = radius
        self.height = height
        self.x = x
        self.y = y

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


def photo_elastic_response_on_particle(canvas, x_min, y_max, max_radius, particle, forces, f_sigma, pixels_per_radius, cutoff=np.inf):
    '''Function computes photoelastic reponse at each point'''
    pixel_to_coordinate = np.linspace(-particle.radius, particle.radius, 2 * pixels_per_radius)
    radius_sqr = np.square(particle.radius)
    xy = []
    for i in np.arange(2 * pixels_per_radius):
        for j in np.arange(2 * pixels_per_radius):
            if np.square(pixel_to_coordinate[i]) + np.square(pixel_to_coordinate[j]) < radius_sqr:
                x = int(np.ceil((particle.x + pixel_to_coordinate[i] - x_min)*pixels_per_radius/max_radius))
                y = int(np.ceil((y_max - particle.y - pixel_to_coordinate[j])*pixels_per_radius/max_radius))
                canvas[x, y] = photo_elastic_response_at_xy(pixel_to_coordinate[i], pixel_to_coordinate[j], particle, forces, f_sigma, cutoff)
    print(xy)
    return canvas

def create_canvas(df_xy, pixels_per_radius):
    max_radius = df_xy['r'].max()
    x_min, x_max,  = df_xy['x'].min()-1.5*max_radius, df_xy['x'].max()+1.5*max_radius,
    y_min, y_max = df_xy['y'].min()-1.5*max_radius, df_xy['y'].max()+1.5*max_radius
    canvas = np.zeros((int(np.ceil((x_max-x_min)*pixels_per_radius/max_radius)), 
                      int(np.ceil((y_max-y_min)*pixels_per_radius/max_radius))))
    return canvas, x_min, y_max, max_radius
    
def angle_finder(cos_val, sin_val):
    '''This function gives the value of the angle in radians from 
    the sin and cos values'''
    if sin_val >= 0 and cos_val >= 0:
        true_angle = asin(sin_val)
    elif sin_val >= 0 and cos_val < 0:
        true_angle = acos(cos_val)
    elif sin_val < 0 and cos_val < 0:
        true_angle = -asin(sin_val)+pi
    elif sin_val < 0 and cos_val >= 0:
        true_angle = asin(sin_val) + 2*pi
    return true_angle

if __name__ == '__main__':
    
    n_pixels_per_radius = 28
    actual_f_sigma = 1
    Cutoff = 10 
    height = 1
    
    path_to_table = os.path.join(os.getcwd(), 'table.csv')
    df_xy = pd.read_csv(path_to_table, header = None, names = ['num', 'x', 'y', 'r'])
    print(df_xy)
    canvas, x_min, y_max, max_radius = create_canvas(df_xy, n_pixels_per_radius)
    print(canvas.shape)
    for i in df_xy['num']:
        particle = Particle(df_xy[df_xy['num'] == i]['x'].values[0],
                            df_xy[df_xy['num'] == i]['y'].values[0],
                            df_xy[df_xy['num'] == i]['r'].values[0],
                            height)
        F = 0.2 #[0.01, 0.4]
        f1 = Force(F,  2, 0)
        f2 = Force(F, 2+np.pi, 0)
        all_forces = [f1, f2]
        canvas  = photo_elastic_response_on_particle(canvas, x_min, y_max, max_radius, 
                                                     particle, all_forces, actual_f_sigma, n_pixels_per_radius)
        
    
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(np.asarray(canvas), vmin=-0, vmax=1, cmap='gray')
    plt.show()
    plt.close()
        
    
    #multiprocessing with 4 processes
    #num_processes = cpu_count()
    #pool = Pool(processes = num_processes)
    
    
        
    
