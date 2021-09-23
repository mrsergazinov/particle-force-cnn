import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import Sequence
from scipy.ndimage.filters import gaussian_filter

class GaussBlur:
    '''Defines gaussian blur class'''
    def __init__(self, radius):
        self.radius = radius
    def blur(self, image):
        return gaussian_filter(image, sigma = self.radius)

class DataGenerator:
    ''' Generates augemented data for Keras using the preprocessing function'''
    def __init__(self, list_image_paths,  
                 dim,
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
            image = image.resize(self.dim, resample = Image.NEAREST)
            image = np.array(image)[:, :, :self.n_channels]
            image = image * self.rescale
            if self.preprocessing_func is not None:
                image = self.preprocessing_func(image)
            # Store sample
            X[i,] = image

        return X
    
    def save(self, path):
        ''' Saves the images resized and preprocessed into a specified location '''

        # Save data
        for i, image_path in enumerate(self.list_image_paths):
            # Load image and transform
            image = Image.open(os.path.join(image_path))
            image = image.resize(self.dim, resample = Image.NEAREST)
            image = np.array(image)[:, :, :self.n_channels]
            image = image * self.rescale
            if self.preprocessing_func is not None:
                image = self.preprocessing_func(image)
            # Store sample
            image = Image.fromarray(image)
            image.save(os.path.join(path, "img" + str(i) + ".jpg"))

def sorter(item):
    ''' Define sorter of image names in order by image number (default is alphanumeric) '''
    radius = float(item[1 : item.find('_')])
    num_img = int(item[item.find('g') + 1 : item.find('j') - 1])
    return (radius, num_img)

class DataGeneratorTrain(Sequence):
    'Generates data for Keras'
    def __init__(self, list_image_paths = None, labels = None,  
                 batch_size = 32, dim = None, n_channels = 3, rescale = 1, 
                 shuffle=True, save_dir = None, preprocessing_func = None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_image_paths = list_image_paths
        self.n_channels = n_channels
        self.rescale = rescale
        self.shuffle = shuffle
        self.save_dir = save_dir
        self.preprocessing_func = preprocessing_func
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_image_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indices)
        
        if self.save_dir is not None:
            for i in range(X.shape[0]):
                path = os.path.join(self.save_dir, 'img' + str(i) + '.jpg')
                plt.imsave(path, np.asarray(X[i, ]), vmin = 0, vmax = 1)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.list_image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        # Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        list_image_paths_batch = [self.list_image_paths[k] for k in indices]
        
        # Get labels
        y = np.array([self.labels[k, :] for k in indices])
        
        # Generate data
        for i, image_path in enumerate(list_image_paths_batch):
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

        return X, y
