#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:53:34 2020

Functions used in time series linking after getting leaf instance segmentation result


@author: hudanyunsheng
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import random
import math
import skimage.io
import pickle as pkl
import re
from skimage.measure import find_contours
from matplotlib import patches, lines
from matplotlib.patches import Polygon
from plantcv import plantcv as pcv
import copy
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Directory of library packages. Note: you will have to change this to the directory of your maskRCNN package
LIB_DIR  = '/shares/mgehan_share/hsheng/Mask_RCNN'
# from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config 
from mrcnn import utils 
from mrcnn import model as modellib 
from mrcnn import visualize 

# Direcotry of dataset used in training. Modify it according to your path 
dataset_dir ='/mnt/efs/data/synthetic_arabidopsis_LSC'
num_images, Image_IDs_train, Image_IDs_val, Image_IDs_test = generate_IDs(dataset_dir)


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


def _get_ax(rows=1, cols=1, size=16):   #???
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax

def _random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


## Set Hyperparameter for Training 
class LeavesConfig(Config):
    '''Configuration for training on the Synthetic Arabidopsis dataset. 
    Derives from the base Config class and overrides values specific to 
    the leave dataset 
    '''
    
    # Give the configuration a recognizable name  
    NAME = 'leaves'
    
    # Number of classes(including background)
    NUM_CLASSES = 1 + 1 # background + leaves 
    
    # Train on 1 GPU AND 5 images per GPU. We can put multiples images on each 
    # GPU because the images are samll. Batch size is 5 (GPU * images/GPU)
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 4  # Modify according to your GPU memory. We trained this on AWS P2
    Batch_size = GPU_COUNT * IMAGES_PER_GPU
    
    # Number of training and validation steps per epoch 
    STEPS_PER_EPOCH = (num_images * 0.6)// Batch_size  # define dataset_IDS
    VALIDATION_STEPS = max(0, (num_images * 0.2) // Batch_size) 
    
    # Don't exclude based on confidence. ##??
    DETECTION_MIN_CONFIDENCE = 0.8
    DETECTION_NMS_THRESHOLD = 0.48
    
    # Backbone network architecture 
    # Supported values are: resnet50, resnet101
    BACKBONE = 'resnet50'
    
    # Input image resizing 
    # Random crops of size 512x512 
    IMAGE_RESIZE_MODE = 'crop'
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0 
    
    # Length of square anchor side in pixels.
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)  
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9    

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])  

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True  
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 150

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50   ## ?? you can adjust this to smaller number

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 100  # ?? this number can be much less


##  Set Hyperparameter for Testing 
    
class LeavesInferenceConfig(LeavesConfig):
    # Set batch size to 1 to run and inference one image at a time 
    GPU_COUNT = 1 
    IMAGES_PER_GPU =1 
    # Don't resize image for inferencing 
    IMAGE_RESIZE_MODE = 'pad64'
    # Non-max suppression threhold to filter RPN proposals 
    # You can increase this during training to generate more proposals
    RPN_NMS_THRESHOLD = 0.9 



class instance_seg_inferencing():
#     def __init__(self, imagedir, savedir, rootdir, pattern_datetime, suffix, id_plant, class_names=['BG', 'Leaf']):
    def __init__(self, imagedir, savedir, rootdir, pattern_datetime, suffix, class_names=['BG', 'Leaf']):
        
        # Directory of original images 
        self.imagedir    = imagedir 

        # Directory of results saving
        junk         = datetime.datetime.now()
        subfolder    = '{}-{}-{}-{}-{}'.format(junk.year, str(junk.month).zfill(2), str(junk.day).zfill(2), str(junk.hour).zfill(2), str(junk.minute).zfill(2))
        self.savedir = os.path.join(savedir, subfolder)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        # Directory of instance segmentation 
        self.segmentation_dir = os.path.join(self.savedir, 'segmentation')
        if not os.path.isdir(self.segmentation_dir):
            os.makedirs(self.segmentation_dir)

        self.rootdir = rootdir

        self.pattern_datetime = pattern_datetime

        self.suffix = suffix

        self.class_names = class_names

    def get_configure(self):
        if not os.path.exists(os.path.join(self.savedir, 'parameters.pkl')):
            self.config = LeavesInferenceConfig()
            parameters = dict()
            parameters['mrcnn_config'] = self.config
            parameters['data']         = self.imagedir
            pkl.dump(parameters, open(os.path.join(self.savedir, 'parameters.pkl'), 'wb'))
        else:
            parameters = pkl.load(open(os.path.join(self.savedir, 'parameters.pkl'), 'rb'))
            self.config = parameters['mrcnn_config']

    def load_model(self):
        # Create model in inference mode
        with tf.device("/cpu:0"):
            self.model = modellib.MaskRCNN(mode= "inference" ,
                                      model_dir=self.rootdir,   #???
                                      config=self.config)
            
        ## load pre-trained weights
        weights_name = os.path.join(self.rootdir, 'model.h5')
        print("Loading weights ", weights_name)
        self.model.load_weights(weights_name, by_name=True)

    def define_colors(self):
        ## generate colors to use in visualization all the time, and save the colors
        if not os.path.exists(os.path.join(self.savedir , 'colors.pkl')):
            self.colors = _random_colors(50)
            pkl.dump(self.colors, open(os.path.join(self.savedir , 'colors.pkl'), 'wb'))
        else:
            self.colors = pkl.load(open(os.path.join(self.savedir , 'colors.pkl'), 'rb'))

    def get_file_list(self):
    ## Get the list of all files
        # suffix = '_crop-img{}.jpg'.format(id_plant)
        # suffix = '_crop.jpg'

        self.list_f = [f for f in os.listdir(self.imagedir) if f.endswith(self.suffix)]
        self.list_f.sort()
        print('There are {} images.'.format(len(self.list_f)))

    def segmentation_inferencing(self, filename):
        temp       = re.search(self.pattern_datetime, filename)
        timepart   = temp.group()
        img        = skimage.io.imread(os.path.join(self.imagedir, filename))

        # Run detection
        results = self.model.detect([img], verbose=1)
        r = results[0]

        # Visualize results
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                    self.class_names, r['scores'], ax=_get_ax(rows=1, cols=1, size=16),show_bbox=True, show_mask=True,
                                    title="Instance Segmentation", colors=self.colors)

        plt.savefig(os.path.join(self.segmentation_dir, timepart + '.png'))
        pkl.dump(r, open(os.path.join(self.segmentation_dir, timepart + '.pkl'), 'wb'))

        masks = r['masks']
        rois  = r['rois']
        F_DIR = os.path.join(self.segmentation_dir, 'masks', timepart)
        if not os.path.isdir(F_DIR):
            os.makedirs(F_DIR)
            
        # show separate visualization
        for idx in range(0, r['masks'].shape[2]):
            mask_i     = np.expand_dims(masks[:,:,idx], 2)
            roi_i      = np.expand_dims(rois[idx], 0)
            class_id_i = np.expand_dims(r['class_ids'][idx],0)
            score_i    = np.expand_dims(r['scores'][idx],0)
            visualize.display_instances(img, roi_i, mask_i, class_id_i, 
                                self.class_names, score_i, ax=_get_ax(rows=1, cols=1, size=16),show_bbox=True, show_mask=True,
                                title="Leaf {}".format(idx), colors = self.colors[idx:idx+1])
            plt.savefig(os.path.join(F_DIR, 'leaf_{}.png'.format(idx)))

    def inferencing_random_sample(self):
        # file_names = next(os.walk(self.imagedir))[2]
        file_name  = random.choice(self.list_f)
        self.segmentation_inferencing(file_name)

    def inferencing_all(self):
        count = 0
        for filename in self.list_f:
            self.segmentation_inferencing(filename)
            count += 1
            print('{} images done. The last one is {}'.format(count, filename))

