"""
This file contains code to generate a list of forces acting on a circular particle, given the appropriate parameters for 
the diversity of forces that we want to see in the output. Moreover, all the generated force lists 
are going to obey the requirments postulated for the uniformly-sized particles.
"""

import numpy as np
from scipy.stats import truncexpon
from math import sin, cos, asin, acos, pi, sqrt
from classes import *


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

def calculate_last_force(F):
    '''Given list of forces, the function outputs the last force, based on balance equations: 
        for tangential and inner components'''

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
    f_last.phi = (angle_finder(cos_val, sin_val) + pi) % (2 * pi)
    f_last.alpha = asin(tang_component / f_last.get_mag())

    return f_last

def calculate_conv_angle(angle):
    if angle > pi:
        return (2 * pi - angle)
    else:
        return angle
        
def list_of_force_angle_lists(num_forces, num_mags, num_angles_tang, num_angles_inner, f_lower_bound, f_upper_bound, delta_angle_inner):
    '''
    This function generates force lists.
    
    At most it will generate num_mags * num_angles_tang * num_angles_inner * num_random 
    force lists.
    
    The process works as follows:
        1. The magnitude value gets sequentially drawn from an equally-spaced sequence 
        from f_lower_bound to f_upper_bound that consists of num_mags elements
        2. Then the following process is repeated num_angles_tang times:
            i. (num_forces - 1) forces are generated sequentially:
                a. The magnitude comes from a random normal distribution with mean = mag
                and std. dev = mag / 5
                b. The tangential component is generated by random normal with mean = 
                0 and std.dev = pi/12, which is to ensure that the tangential anle is mostly
                within pi/4 and -pi/4
                c. The interval [0, 2*pi] is divided into num_forces equal sub-intervals, 
                such that these sub-intervals are delta_angle_inner apart. The position angle for each 
                of the forces is drawn from uniform distribition over the correspodning 
                sub-interval.
            ii. The last force is calculated to ensure that the balance equations are satisfied
            iii. The forces are checked again to ensure that they are at least delta_angle_inner apart
            and each has the tangential component within -pi/4 to pi/4. This is necessary as
            the last force may have changed this. 
            iv. If the conditions are satisfied then we have a ready-to-go force list. We
            repetitively (num_angles_inner times) add this force list to the final list of force lists as follows:
                a. At each step, we shift all of the position angles by 2*pi / (num_forces * num_angles_inner).
                b. We attach num_random copies of the current force list to the final list of 
                force lists.
    
    In total, at most (num_random * num_angles_inner * num_angles_tang * num_mags) force lists corresponding to 
    (physically feasible) particles are generated. The number of force lists may be smaller if at some iteration of the loops,
    we cannot generate physically feasible list for more than max_attempts = 10^6 attempts. 
    '''
    list_of_F_lists = []
    
    # delta serves to ensure that the sub-inrervals for position angles are pi/3 apart
    # epsilon is defined to shift the angles while attached the force list to the final list of 
    # force lists
    # phi_init is defined to help produce the sub-intervals for position angles from the interval
    # [0, 2*pi]
    epsilon = 2 * pi / (num_forces * num_angles_inner)
    phi_init = 0  
    
    # alpha_init and alpha_std_dev are parameters for the random normal distribution to produce
    # tangential angles
    delta = delta_angle_inner / 2
    alpha_init = 0
    alpha_std_dev = pi/12
    # the std. dev was chosen so that alpha mostly stays within -pi/4 to pi/4, which is
    # needed because |f_tang| < |f_inner| due to physical constraints
    
    # max_attempts is defined to limit the number of attempts to produce a force list given a
    # starting magnitude. This is done to speed up computations. 
    max_attempts = 10**6
    
    # draw mean force
    scale = 1 / 2
    rv = truncexpon(b = (f_upper_bound-f_lower_bound) / scale, loc = f_lower_bound, scale = scale)
    mags = rv.rvs(num_mags)
    for mag in mags:
        for i in range(num_angles_tang):
            # count attempts made to generate new force list
            attempt_count = 0
            while attempt_count < max_attempts:
                F_list = []
                for j in range(num_forces-1):
                    # generate (num_forces - 1) forces acting on particle
                    f_mag = abs(np.random.normal(mag, mag/5))
                    f_phi = np.random.uniform(phi_init + j * shift + delta,
                                            phi_init + (j + 1) * shift - delta) % (2 * pi)
                    f_alpha = np.random.normal(alpha_init, alpha_std_dev)
                    F_list.append(Force(f_mag, f_phi, f_alpha))
                # calculate last force to balance the rest
                f_last = calculate_last_force(F_list)
                F_list.append(f_last)
                
                # check that the generated forces are physically feasible
                check = [calculate_conv_angle((f_last.get_phi() - f.get_phi()) % (2 * pi)) >= delta_angle_inner for f in F_list[:-1]]
                check += [(f.get_alpha()<=pi/4 and f.get_alpha()>=-pi/4) for f in F_list]
                check += [(f.get_mag() >= 0) for f in F_list]
                attempt_count+=1
                
                if all(check):
                    for ang_inner in range(num_angles_inner):
                        # add num_angles_inner rotations
                        F_list_new = [Force(f.get_mag(), (f.get_phi() + ang_inner * epsilon) % (2 * pi), f.get_alpha())
                                      for f in F_list]
                        list_of_F_lists.append(F_list_new)
                    attempt_count = max_attempts
    return list_of_F_lists
