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

# Import Leaf config
LEAF_DIR = os.path.join(ROOT_DIR, 'Leaf') #'/shares/mgehan_share/hsheng/projects/maskRCNN/InstanceSegmentation/Leaf'
sys.path.append(LEAF_DIR)
import Leaf

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
        # leaf_dir = os.path.join(self.rootdir, 'Leaf')
        # sys.path.append(leaf_dir)
        # import Leaf

        self.pattern_datetime = pattern_datetime

        self.suffix = suffix

#         self.id_plant = id_plant

        self.class_names = class_names

    def get_configure(self):
        if not os.path.exists(os.path.join(self.savedir, 'parameters.pkl')):
            self.config = Leaf.LeavesInferenceConfig()
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
        weights_name = os.path.join(self.rootdir, 'mask_rcnn_leaves_0060.h5')
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


