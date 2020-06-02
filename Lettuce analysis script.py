import skimage
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from collections import OrderedDict


import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

image = imread('Trial 6 early lettuce-correct.jpg')

def get_parameters(h):
    ''' Takes 2d image and returns peak parameters
        This was optimised for the lettuce image
    '''
    h = h
    # Having a relativley small number of bins makes it easier to find distributions
    bins = 50
    #halves the peak so we don't get noisy edges
    trim_factor = 2
    # find the peaks
    hist, bin_edges = np.histogram(h,bins)
    
    #find the threshold parameters, returns the two largest but i am only using the largest atm
    peaks_dict = dict(zip(hist, bin_edges))
    d_descending = OrderedDict(sorted(peaks_dict.items(), key=lambda kv: kv[0]))
    prominence_val = round(list(d_descending.keys())[-1]* 0.5)
    peaks, properties = find_peaks(hist, width = 1, prominence = prominence_val)
    peak_parameters = (properties['right_bases'][0] / bins)/trim_factor, (properties['left_bases'][0] / bins)*trim_factor

    return peak_parameters

def clean_image(image_c):
    ''' Takes 2d array and does an otsu to clear around feature'''
    #clean the image
    #use the green channel
    g = image_c[:,:,1]
    thresh = threshold_otsu(g)
    # Fill in the gaps
    bw = closing(g > thresh, square(3))
    #return masked image
    image_c[~bw] = [0,0,0]
    
    return image_c
    

def get_leaves(image):
    ''' Takes and rgb image and returns a masked copy'''
    image_c = np.copy(image)
    image_hsv = rgb2hsv(image_c )
    h = image_hsv[:,:,0]
    
    peak_parameters = get_parameters(h)
    mask = (h < peak_parameters[0]) & (h > peak_parameters[1])
    
    #mask the image
    image_c[~mask] = [0,0,0]
    
    image_c = clean_image(image_c)

    return image_c 

    
plt.imshow(get_leaves(image))
