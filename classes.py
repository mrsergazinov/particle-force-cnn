"""
This file defines Particle and Force classes needed for data handling and manipulation.
"""

import numpy as np

class Particle:
    def __init__(self, radius=1, height=0.1):
        self.radius = radius
        self.height = height

class Force:
    """
        Each force acting on this particle at its boundary is described by three variables 
    """
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