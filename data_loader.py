import numpy as np
import os
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

class GaussBlur:
    '''Defines gaussian blur class'''
    def __init__(self, radius):
        self.radius = radius
    def blur(self, image):
        return gaussian_filter(image, sigma = self.radius)

class DataGenerator:
    ''' Generates augemented data for Keras using the preprocessing function'''
    def __init__(self, list_image_paths = None,  
                 dim = None,
                 n_channels = 3, 
                 rescale = 1, 
                 preprocessing_func = None):
        'Initialization'
        self.list_image_paths = list_image_paths
        self.dim = dim
        self.n_channels = n_channels
        self.rescale = rescale
        self.preprocessing_func = preprocessing_func

    def generate(self):
        ''' Generates data containing batch_size samples'''
        
        # Initialisation
        X = np.empty((len(self.list_image_paths), *self.dim, self.n_channels))
        
        # Generate data
        for i, image_path in enumerate(self.list_image_paths):
            # Load image and transform
            image = Image.open(os.path.join(image_path))
            if self.dim is not None:
                image = image.resize(self.dim, resample = Image.NEAREST)
            image = np.array(image)[:, :, :self.n_channels]
            image = image * self.rescale
            if self.preprocessing_func is not None:
                image = self.preprocessing_func(image)
            # Store sample
            X[i,] = image

        return X

def sorter(item):
    ''' Define sorter of image names in order by image number (default is alphanumeric) '''
    radius = float(item[1 : item.find('_')])
    num_img = int(item[item.find('g') + 1 : item.find('j') - 1])
    return (radius, num_img)
