#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:48:34 2019

@author: wzhan
"""

import os 
import random


def generate_IDs(dataset_dir):
    '''Generate the list to load. Because the dataset is all in one directory. 
    We assume the filename of image is like 'plant00000_rgb.png' because I used 
    x[0:10] in line 31. You can modify according to your filename.  We need to do 
    train/validation/test split with ratio of 6:2:2. 
    Input: 
    dataset_dir: the path where stores the dataset 
    Output: 
    num_images: the total number of images in the dataset_dir 
    Image_IDs_train: A list with length of int(0.6 * num_images). The element is 
    a str (like plant00003). 
    Image_IDs_val: A list with length of int(0.2 * num_images). The element is 
    a str (like plant00003)
    Image_IDs_test: A list with length of int(0.2 * num_images). The element is 
    a str (like plant00003)
    
    '''
    AllImage_IDs = []
    for (root, dirs, filenames) in os.walk(dataset_dir):
        for x in filenames:
            if x.endswith('rgb.png'):
                AllImage_IDs.append(x[0:10])
    num_images = len(AllImage_IDs)
    Image_IDs_train = random.sample(AllImage_IDs, int(0.8 * num_images))
    Image_IDs_test = list(set(AllImage_IDs) - set(Image_IDs_train))
    Image_IDs_val = random.sample(Image_IDs_train, int(0.2 * num_images))
    Image_IDs_train = list(set(Image_IDs_train) - set(Image_IDs_val))
    
    return num_images, Image_IDs_train, Image_IDs_val, Image_IDs_test
    
    

    
