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

def calculate_last_force(F, alpha_init, alpha_std_dev):
    '''Given list of forces, the function outputs the last force, based on balance equations'''

    f_last = Force()
    
    x_component_inner, y_component_inner = 0, 0
    inner_magnitude = 0
    tang_component = 0
    
    for f in F:
        x_component_inner -= f.get_mag()*cos(f.get_alpha())*cos(f.get_phi()+pi)
        y_component_inner -= f.get_mag()*cos(f.get_alpha())*sin(f.get_phi()+pi)
        tang_component -= f.get_mag()*sin(f.get_alpha())
        
    inner_magnitude = sqrt(x_component_inner**2 + y_component_inner**2)
    cos_val = x_component_inner / inner_magnitude
    sin_val = y_component_inner / inner_magnitude
    
    f_last.magnitude = sqrt(inner_magnitude**2 + tang_component**2)
    f_last.phi = (angle_finder(cos_val, sin_val)+pi)%(2*pi)
    f_last.alpha = asin(tang_component/f_last.get_mag())

    return f_last

def check_balance(F):
    '''Given list of forces, the function outputs the last force, based on balance equations'''
    x,y = 0,0
    for f in F:
        x += f.get_mag()*cos(f.direction_angle_xy)
        y += f.get_mag()*sin(f.direction_angle_xy)
    print(x, y)
    return None
        
def list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, num_random, f_lower_bound, f_upper_bound):
    '''This function generates list of lists of forces,
    such that the lists have num_forces forces, num_mags different force magnitudes, num_angles different angles, and
    each list is repeated num_random time'''
    list_of_F_lists = []
    
    delta = pi/6
    shift = 2 * pi / num_forces
    epsilon = pi/(2*num_angles_inner)
    
    phi_init = 0   
    alpha_init = 0
    alpha_std_dev = pi/12
    #the std. dev was chosen so that alpha stays within -pi/4 to pi/4, which is
    #needed because |f_tang| < |f_inner| due to physical constraints
    
    max_attempts = 10**6
    for mag in np.linspace(f_lower_bound, f_upper_bound, num_mags):
        for i in range(num_angles_tang):
            attempt_count = 0
            while attempt_count < max_attempts:
                F_list = []
                for i in range(num_forces-1):
                    f_mag = abs(np.random.normal(mag, mag/5))
                    f_phi = np.random.uniform(phi_init+i*shift+delta, 
                                            phi_init+i*shift+shift-delta)%(2*pi)
                    f_alpha = 0
                    #np.random.normal(alpha_init, alpha_std_dev)
                    F_list.append(Force(f_mag, f_phi, f_alpha))
                f_last = calculate_last_force(F_list, alpha_init, alpha_std_dev)
                F_list.append(f_last)
                check = [abs(f_last.get_phi() - f.get_phi()) >= pi/3 for f
                         in F_list[:-1]] + [(f.get_alpha()<=pi/4 and f.get_alpha()>=-pi/4) for f in F_list]
                attempt_count+=1
                
                if all(check):    
                    for ang_inner in range(num_angles_inner):
                        F_list_new = [Force(f.get_mag(), (f.get_phi()+ang_inner*epsilon)%(2*pi), f.get_alpha())
                                      for f in F_list]
                        for rand in range(num_random):
                            list_of_F_lists.append(F_list_new)   
                    attempt_count = max_attempts
    
    return list_of_F_lists

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
    mu = np.max(img)
    sigma = np.std(img)
    for ix,iy in np.ndindex(img.shape):
        if (img[ix,iy] != 0): 
            img[ix,iy] += coef_random*np.random.normal(mu, sigma)
    return img

if __name__ == '__main__':
    #max/min magnitude for force
    lower_bound = 0.01
    upper_bound = 0.4
    #number of randomly noised pictures for each force selection
    num_random = 1
    #max/min number of forces
    max_num_forces = 6
    min_num_forces = 2
    #number of different magnitudes for forces
    num_mags = 3
    #number of different angles 
    num_angles_inner = 4
    num_angles_tang = 1
    #num pixles in image *2
    n_pixels_per_radius = 28
    #num forces
    num_forces = 4
    
    #multiprocessing with 4 processes
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
        fig.add_subplot(6, 3, i+1)
        plt.imshow(np.asarray(data[i,:,:]), vmin= 0, vmax = 1, cmap='gray')
        
    np.save(os.path.join(os.getcwd(), 'data_test'), data)
        
# =============================================================================
#     #create matrix that will store images
#     data = np.empty((0, n_pixels_per_radius*2, n_pixels_per_radius*2))
#     #create storage matrices for labels
#     labels_angles = dict()
#     labels_mags = dict()
#     labels_num_forces = np.array([])
#     
#     #generate data
#     for num_forces in range(min_num_forces, max_num_forces+1):
#         start_time = time.time()
#         list_of_F = list_of_force_angle_lists(num_forces, num_mags, num_angles, num_random, lower_bound, upper_bound)
#         labels_num_forces = np.concatenate((labels_num_forces,
#                                             np.zeros(len(list_of_F)) + num_forces))
#         labels_angles[num_forces] = [[f.get_phi() for f in F] for F in list_of_F]
#         labels_mags[num_forces] = [[f.get_mag() for f in F] for F in list_of_F]
#         d = np.array(pool.map(image_gen, list_of_F))
#         data = np.concatenate((data, d), axis = 0)
#         print("--- %s seconds ---" % (time.time() - start_time))
# 
# =============================================================================  
        
        
    
