import os
from os import path
from datetime import datetime
import pathlib
import numpy as np
from collections import OrderedDict, Counter

#Image stuff
import skimage
from skimage.io import imread
from skimage.filters import threshold_otsu, sobel, gaussian
from skimage.segmentation import clear_border, watershed
from skimage.color import rgb2hsv, hsv2rgb
from skimage.measure import label, regionprops
from skimage.morphology import closing, square


from scipy.signal import find_peaks
from scipy.misc import electrocardiogram
import scipy.ndimage as ndi
from scipy.signal import find_peaks
from scipy import stats


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


import csv
from tqdm import tqdm


#Gets current path
path =  pathlib.Path().absolute()

#Use if need to be explicit
#path = path.Path(r'')
temp_files = os.listdir(path)

def get_timestamp(file):
    #is this indented?
    
    for file in files:
        numbers = file.split('/')
        timestamp = numbers[-1][:-4]

    return timestamp

#this loop creates a list of all the images and ignores any other files that are not jpgs
files = []
for x in temp_files:
    if x.endswith('.png'):
        files.append(x)


#timestamp = get_timestamp(file)

def convert_to_day(timestamp):

    date = datetime.fromtimestamp(float(timestamp))
    #month-day_hour_minute
    name = datetime.strftime(date,'%m-%d_%H_%M')


    return name

def get_parameters(h):
    
    ''' Takes 2d image and returns peak parameters
        This was optimised for the lettuce image
    '''
    h = h
    # Having a relativley small number of bins makes it easier to find distributions
    bins = 50
    #trims the peak so we don't get noisy edges, this is a nice balance, but 2 works best from most images
    #however, it cuts off some images too much.
    trim_factor = 1.9#3.2
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

# Some images are noisy around the edges this helps..but adds alot more time.

def get_leaves_lettuce(image):
    ''' Only diff between this and Rocket is the #2 edge size'''
    hsv = rgb2hsv(image)
    s = hsv[:,:,1]
    sobel_edge = sobel(gaussian(s, 4))

    markers = np.zeros_like(sobel_edge)

    markers[sobel_edge < 0.0000001] = 1 #green
    markers[sobel_edge > 0.001] = 2 # edges white 0.004 for lettuce 0.00037

    segmented = watershed(sobel_edge, markers)
    # not sure what this does
    segmented = ndi.binary_fill_holes(segmented - 1)
    copied_image = np.copy(image)
    mask = skimage.morphology.remove_small_objects(segmented, 100000)
    copied_image[~mask] = [0,0,0]

    return copied_image

def make_stats_dict(masked_img):
    hsv = rgb2hsv(masked_img) #cv2.cvtColor(masked_img, cv2.COLOR_RGB2HSV)

    h = hsv[:,:,0]

    h  = np.around(h * 180, 0)

    #no idea why I have to do this twice each time
    index = np.argwhere(h==0)
    h = np.delete(h,index)

    index = np.argwhere(h==0)
    h = np.delete(h,index)

    c = Counter(h)

    hue_median = np.median(h[np.where(h > 0)])
    hue_circular_mean = stats.circmean(h[np.where(h > 0)], high=179, low=0) # low used to be 0
    hue_circular_std = stats.circstd(h[np.where(h > 0)], high=179, low=0) 
    hue_range = len(np.unique(h))

    stats_dict = {'median':hue_median, 'mean':hue_circular_mean, 
                  'std':hue_circular_std, 'range':hue_range, 'hue_list':h,
                   'counter':c}

    return stats_dict

def total_colour(c,colour):
    c =  c
    colour = colour
    number = 0
    for i in colour:
        number += c[i]
    return number

def get_hues(counter):


    ordered = dict(sorted(counter.items(), key=lambda t: t[0]))
    possible_values = list(range(1,181))

    #this loop adds in mising values
    for i in possible_values:
        if i not in ordered.keys():
            ordered.update({i:0})


    return ordered # return the list 

def get_colour_dict(stats_dict):

    red = range(0,8)
    orange = range(8,16)
    yellow = range(16,31)
    green = range(31,76)
    cyan = range(76,106)
    blue = range(106,128)
    violet = range(128,143)
    magenta = range(143,165)
    red2 = range(165,180)

    colours = [red,orange,yellow,green,cyan,blue,violet,magenta,red2]

    total = sum(stats_dict['counter'].values())

    stats_dict['red'] = (total_colour(stats_dict['counter'], red) / total)*100
    stats_dict['orange'] = (total_colour(stats_dict['counter'], orange) / total *100)
    stats_dict['yellow'] = (total_colour(stats_dict['counter'], yellow) / total *100)
    stats_dict['green'] = (total_colour(stats_dict['counter'], green)/ total *100)
    stats_dict['cyan'] = (total_colour(stats_dict['counter'], cyan)/ total *100)
    stats_dict['blue'] = (total_colour(stats_dict['counter'], blue)/ total *100)
    stats_dict['violet'] = (total_colour(stats_dict['counter'], violet)/ total *100)
    stats_dict['magenta'] = (total_colour(stats_dict['counter'], magenta)/ total *100)
    stats_dict['red2'] = (total_colour(stats_dict['counter'], red2)/ total *100)

    return stats_dict
    
def clean_up(stats_dict,file_name):
    ''' makes final dict '''

    final = stats_dict
    hues = get_hues(final['counter'])

    od = OrderedDict(sorted(hues.items()))
    del final['counter']
    del final['hue_list']

    #i should be the same as the loop that is going through the images
    final['name'] = file_name

    return final, hues

def total_colour(c,colour):
    c =  c
    colour = colour
    number = 0
    for i in colour:
        number += c[i]
    return number

def make_stats_dict(masked_img):
    hsv = rgb2hsv(masked_img) #cv2.cvtColor(masked_img, cv2.COLOR_RGB2HSV)

    h = hsv[:,:,0]

    h  = np.around(h * 180, 0)

    #no idea why I have to do this twice each time
    index = np.argwhere(h==0)
    h = np.delete(h,index)

    index = np.argwhere(h==0)
    h = np.delete(h,index)

    c = Counter(h)

    hue_median = np.median(h[np.where(h > 0)])
    hue_circular_mean = stats.circmean(h[np.where(h > 0)], high=179, low=0) # low used to be 0
    hue_circular_std = stats.circstd(h[np.where(h > 0)], high=179, low=0) 
    hue_range = len(np.unique(h))

    stats_dict = {'median':hue_median, 'mean':hue_circular_mean, 
                  'std':hue_circular_std, 'range':hue_range, 'hue_list':h,
                   'counter':c}

    return stats_dict

def write_to_csv(final_dict, hues):
    # this adds each hue value as a dict entry
    for k,v in hues.items():
        final_dict[k] = v


    with open('mycsvfile.csv', 'a') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, final_dict.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(final_dict)


for file in tqdm(sorted(files)):
    
    
    file_name = file#convert_to_day(timestamp)  


    image = imread(file)
    

    img = get_leaves(image)
    masked_img = get_leaves_lettuce(img)
    
    stats_dict = make_stats_dict(masked_img)
    get_colour_dict(stats_dict)
    

    final_dict, hues = clean_up(stats_dict,file_name)
    
    file_exists = os.path.isfile('mycsvfile.csv')
    
    
        
            
            
        
    write_to_csv(final_dict, hues)
