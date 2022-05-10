import numpy as np
import cv2
from skimage.feature import hog



class FeatureExtraction(object):
    
    def __init__(self, img, colour_space= 'YCrCb', orient_bins = 10, pixels_per_cell = 8, cells_per_block =2):
        
        self.orientation = orient_bins
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        #Convert the image pixels into the desired colour space
        if colour_space != 'RGB':
            if colour_space == 'HSV':
                self.image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif colour_space == 'LUV':
                self.image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif colour_space == 'HLS':
                self.image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif colour_space == 'YUV':
                self.image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif colour_space == 'YCrCb':
                self.image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: self.image = np.copy(img)
            
        #Get the dimensions of the image
        image_shape = self.image.shape
        self.height = image_shape[0]
        self.width = image_shape[1]
        self.depth = image_shape[2]
        
        #compute the hog features for the whole image over each channel
        self.hog_features = []
        for i in range(self.depth):
            
            hog_feat = hog(self.image[:,:,i], orientations = orient_bins, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                          cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=True, visualise=False,
                           feature_vector=False)
            self.hog_features.append(hog_feat)
            
        self.hog_features = np.asarray(self.hog_features)
        
    def colour_hist(self, img, nbins=32):  #bins_range=(0,256)
        #compute the histrogram of colour channels separatley
        channel0_hist = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))
        channel1_hist = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))
        channel2_hist = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))
        
        hist_features = np.concatenate((channel0_hist[0], channel1_hist[0], channel2_hist[0]))
        
        return hist_features
    
    def bin_spatial(self, img, size=(32,32)):
        colour1 = cv2.resize(img[:,:,0], size).ravel()
        colour2 = cv2.resize(img[:,:,1], size).ravel()
        colour3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((colour1, colour2, colour3))
    
    def hog(self, x, y, region_size):
        
        region = (region_size// self.pixels_per_cell) - 1
        
        location_x = x// self.pixels_per_cell
        location_y = y// self.pixels_per_cell
        
        #if (location_x + region) > self.hog_features.shape[2]:
        #    location_x = self.hog_features.shape[2] - region
            
        #if (location_y + region) > self.hog_features.shape[1]:
        #    location_y = self.hog_features.shape[1] - region
            
        return np.ravel(self.hog_features[:, location_y:location_y+region,  location_x:location_x+region, :, :, :])
            
            
        
    def get_features(self, x=0, y=0, region_size=64, spatial=True, colour_hist=True, hog_features=True):
        features = []       
        if spatial:
            spatial_features = self.bin_spatial(self.image[y:y+region_size, x:x+region_size, :])
            features.append(spatial_features)          
       
        if colour_hist:
            colour_hist_features = self.colour_hist(self.image[y:y+region_size, x:x+region_size, :])
            features.append(colour_hist_features)           
       
        if hog_features:
            hog_features = self.hog(x, y, region_size)
            features.append(hog_features)
            
        return np.concatenate(features)
            
