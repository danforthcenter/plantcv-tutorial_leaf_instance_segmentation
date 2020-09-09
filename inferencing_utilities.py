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
import numpy.matlib
from plantcv.plantcv import fatal_error
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Directory of library packages. Note: you will have to change this to the directory of your maskRCNN package
LIB_DIR  = '/shares/mgehan_share/hsheng/Mask_RCNN'
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config 
from mrcnn import utils 
from mrcnn import model as modellib 
from mrcnn import visualize 
import csv

def calculate_overlaps(masks1, masks2):
    """
    Calculate the overlap between two sets of segmentation masks. 
    masks1: (h,w,num1)
    masks2: (h,w,num2)
    The overlaps are evaluated by: 
    abosulate intersection: (num1, num2)
    IOU(intersection over union): (num1, num2)
    IOS (intersection over area of 1st mask): (num1, num2)
    """
    num1 = masks1.shape[2]
    num2 = masks2.shape[2]
    intersecs = np.zeros((num1, num2))
    unions    = np.zeros((num1, num2))     
    area      = np.zeros(num1)
    for idx in range(0, num1):
        maski  = np.expand_dims(masks1[:,:,idx], axis=2)
        masks_ = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        maski_ = np.reshape(maski > .5, (-1, maski.shape[-1])).astype(np.float32)

        intersect = np.dot(masks_.T, maski_).squeeze()
        union     = np.sum(masks_,0) + np.sum(maski_) - intersect
        intersecs[idx,:] = intersect
        unions[idx,:]    = union
        area[idx]        = np.sum(maski)
    areas = numpy.matlib.repmat(area, num1, 1).T
    IOU   = np.divide(intersecs,unions)
    IOS   = np.divide(intersecs,areas)
    return intersecs, IOU, IOS

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
    
# Direcotry of dataset used in training. Modify it according to your path 
dataset_dir ='/mnt/efs/data/synthetic_arabidopsis_LSC'
num_images, Image_IDs_train, Image_IDs_val, Image_IDs_test = generate_IDs(dataset_dir)

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
    DETECTION_NMS_THRESHOLD  = 0.35
    
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
    POST_NMS_ROIS_TRAINING  = 2000
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
    USE_MINI_MASK   = True  
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


class img_instance_segments():
    """
    A class for instance segmentation, an object represents an image with its instance segmentation results and updated instance segmentation results
    """
    def __init__(self, image, class_names):
        # initialize original image
        self.image = image 

        # initialize class_names
        self.class_names = class_names

        # initialize results
        self.segment = None

        # initialize updated results
        self.segment_ = None

        self.issues = dict()
        self.remove_list = []

    def segmentation_quality_control(self):
        """
        Find issues in segmentation and update segmentation result. Currently there are two types of issues defined:
        type 1: two bounding boxes heavily overlap each other. logic: IOU
            If two segmentation masks have a large IOU (>0.5), they are considered to be heavily overlap each other. In this case, the one with smaller score would be removed.
        type 2: if one mask overlaps with two or more other masks, this mask is considered to be redundant and should be removed. logic: IOS
        
        Input: 
        result: segmentation result (dictionary)
        filename 
        Output:
        issues: a dictionary of 2 types of issues
        remove_list: a list of indices of masks to be removed
        result_: updated result (dictionary)
        """
        masks     = copy.deepcopy(self.segment['masks'])
        rois      = copy.deepcopy(self.segment['rois'])
        class_ids = copy.deepcopy(self.segment['class_ids'])
        scores    = copy.deepcopy(self.segment['scores']) 
        intersecs, IOU, IOS = calculate_overlaps(masks, masks)

        # The diagonal values are overlaps with themselves, which means they are all 1. So they should be replaced with 0.
        np.fill_diagonal(intersecs, 0)
        np.fill_diagonal(IOU, 0)
        np.fill_diagonal(IOS, 0)

        # Initialize dictionary of issues
        # issues      = dict()
        # remove_list = []

        # type 1:  heavily overlap each other (logic: IOU)
        type_iss = 1
        name_iss = 'issue{}'.format(type_iss)
        self.issues[name_iss] = dict()
        locs1x, locs1y = np.where(IOU>.5)# any mask that overlap with another one (with IOU > 0.5)
        if len(locs1x) > 0:
            list_issue1 = []
            for (idx,idy) in zip(locs1x,locs1y):
                self.issues[name_iss][idx] = idy
                areax = np.sum(masks[:,:,idx])
                areay = np.sum(masks[:,:,idy])
                remove_idx = [idx,idy][np.argmin([areax,areay])]
                if remove_idx not in self.remove_list:
                    self.remove_list.append(remove_idx)            

        # type 2: overlap with other two (logic: IOS)
        type_iss = 2
        name_iss = 'issue{}'.format(type_iss)
        self.issues[name_iss] = dict()
        locs2x,locs2y = np.where(IOS>=.5) # find those have large enough IOS values
        idxs,counts     = np.unique(locs2y,return_counts=True)
        if len(np.where(counts>1)[0]): # if there are more than one mask that heavily overlap the same mask (with IOS > 0.5)
            if counts.max()>2:
                print('Warning: File {} issue type {}: not handled! \n'.format(filename, type_iss))
            else:
                idx_2 = np.where(counts==2)[0]
                if len(idx_2)>0: # if there are two masks that heavily overlap the same mask
                    for idx in idx_2:
                        idx_mask = idxs[idx]
                        self.issues[name_iss][idx_mask] = locs2x[np.where(locs2y==idx_mask)]
                        if idx_mask not in self.remove_list:
                            self.remove_list.append(idx_mask)
        self.remove_list.sort(reverse=True)
        
        # update results
        for remove_idx in self.remove_list:
            masks     = np.delete(masks, remove_idx, axis = 2)
            rois      = np.delete(rois, remove_idx, axis = 0)
            class_ids = np.delete(class_ids, remove_idx, axis = 0)
            scores    = np.delete(scores, remove_idx, axis = 0)
            
    #     visualize.display_instances(img, rois, masks, class_ids, class_names, scores, ax=funcs._get_ax(rows=1, cols=1, size=16),show_bbox=True, show_mask=True, title=timepart, colors = colors)
    #     plt.savefig(os.path.join(path_updated, timepart + '.png'))
        self.segment_ = dict()
        self.segment_['masks']     = masks
        self.segment_['rois']      = rois
        self.segment_['class_ids'] = class_ids
        self.segment_['scores']    = scores


    def instance_segmentation(self, model):
        """
        A function to get instance segmentation for one image (including quality control and updating resule)
        Input: 
        image (self.image)
        model: pre_trained model
        Output:
        result: result of instance segmentation
        result_: updated result
        issues: issues found in initial result of instance segmentation (if any)
        remove_list: list of indices  of masks to be removed (if any)

        """

        # temp       = re.search(self.pattern_datetime, filename)
        # timepart   = temp.group()
        # img        = skimage.io.imread(os.path.join(self.imagedir, filename))


        # Run detection
        r = model.detect([self.image], verbose=1)
        self.segment = r[0]
        self.segmentation_quality_control()

    def visualize_whole(self, updated=False, colors=None, print_img=False, savename=None):
        """
        Inputs:
        savename: full name (path and name to save the result)
        """
        if colors is None:
            colors = _random_colors(50)
        if print_img is True and savename is None:
            fatal_error("You must provide a savename to save the result!")
        img = self.image
        if updated is False:
            r      = self.segment
            title1 = "Instance Segmentation" 
        elif updated is True:
            r = self.segment_
            title1   = "Updated Instance Segmentation"
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'], ax=_get_ax(rows=1, cols=1, size=16),show_bbox=True, show_mask=True,
                                    title=title1, colors=colors)
        if print_img is True:
            plt.savefig(savename)
            plt.close('all')
    
    def visualize_separate(self, updated=False, colors=None, print_img=False, savepath=None):
        """
        Inputs:
        savepath: path (subfolder) to save the separate visualization
        savepath = os.path.join(savepath_all, 'masks', imgname)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        """
        if print_img is True and savepath is None:
            fatal_error("You must provide a savepath to save the result!")
        if colors is None:
            colors = _random_colors(50)
        img = self.image
        if updated is False:
            r      = self.segment
            title_ = ''
        elif updated is True:
            r = self.segment_
            title_ = 'Updated'
        num_instances = r['masks'].shape[2] 

        for idx in range(0, num_instances):
            mask_i     = np.expand_dims(r['masks'][:,:,idx], 2)
            roi_i      = np.expand_dims(r['rois'][idx], 0)
            class_id_i = np.expand_dims(r['class_ids'][idx],0)
            score_i    = np.expand_dims(r['scores'][idx],0)
            visualize.display_instances(img, roi_i, mask_i, class_id_i, self.class_names, score_i, ax=_get_ax(rows=1, cols=1, size=16),show_bbox=True, show_mask=True,
                                title="{} Leaf {}".format(title_, idx), colors = colors[idx:idx+1])
            if print_img is True:
                plt.savefig(os.path.join(savepath, 'leaf_{}.png'.format(idx)))
                plt.close('all')


class instance_seg_inferencing():
    """
    A class for a bunch of images ready for instance segmentation
    Getting list of images, loading models, calling functions
    """
    def __init__(self, imagedir, savedir, rootdir, pattern_datetime, suffix, class_names=['BG', 'Leaf'], name_model='model.h5', list_files=None):
        
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

        # Directory of updated instance segmentation (after quality control)
        self.updated_dir = os.path.join(self.segmentation_dir, 'updated')
        if not os.path.isdir(self.updated_dir):
            os.makedirs(self.updated_dir)

        self.rootdir = rootdir

        self.pattern_datetime = pattern_datetime

        self.suffix = suffix

        self.class_names = class_names

        self.list_files = list_files

        self.name_model = name_model

    def get_configure(self):
        if not os.path.exists(os.path.join(self.savedir, 'parameters.pkl')):
            self.config = LeavesInferenceConfig()
            parameters  = dict()
            parameters['mrcnn_config'] = self.config
            parameters['data']         = self.imagedir
            pkl.dump(parameters, open(os.path.join(self.savedir, 'parameters.pkl'), 'wb'))
            txtfile = open(os.path.join(self.savedir, 'config.txt'), 'w')
            for att in dir(self.config):
                if not att.startswith('__'):
                    txtfile.write('{}: {}\n'.format(att, getattr(self.config, att)))
            txtfile.close()
        else:
            parameters  = pkl.load(open(os.path.join(self.savedir, 'parameters.pkl'), 'rb'))
            self.config = parameters['mrcnn_config']

    def load_model(self):
        # Fetch model in inference mode
        with tf.device("/cpu:0"):
            self.model = modellib.MaskRCNN(mode= "inference",
                config=self.config,
                model_dir=self.rootdir)

        ## load pre-trained weights
        weights_name = os.path.join(self.rootdir, self.name_model)
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
        if self.list_files is not None:
            self.list_f = self.list_files
        else:
            self.list_f = [f for f in os.listdir(self.imagedir) if f.endswith(self.suffix)]
        self.list_f.sort()
        if len(self.list_f) == 1:
            print('There is only 1 image.')
        elif len(self.list_f) > 1:
            print('There are {} images.'.format(len(self.list_f)))
        else:
            fatal_error('There is no images given the combination conditions of suffix, file list, please double check!')

    def result_visualization(self, img_inst_segs, updated=False, print_img=False, savename=None):
        if print_img is True:
            if updated is False:
                savepath = self.segmentation_dir
            elif updated is True:
                savepath = self.updated_dir
            savename1 = os.path.join(savepath, savename)
            savepath2 = os.path.join(savepath, 'masks', savename)
            if not os.path.isdir(savepath2):
                os.makedirs(savepath2)
        else:
            savename1 = None
            savepath2 = None
        # visualization in whole
        img_inst_segs.visualize_whole(updated=updated, colors=self.colors, print_img=print_img, savename=savename1)
        # separate visualization
        img_inst_segs.visualize_separate(updated=updated, colors=self.colors, print_img=print_img, savepath=savepath2)

    def segmentation_inferencing(self, filename):
        temp       = re.search(self.pattern_datetime, filename)
        timepart   = temp.group()
        img        = skimage.io.imread(os.path.join(self.imagedir, filename))

        # initialize img_instance_segments class
        img_inst_segs  = img_instance_segments(image=img, class_names = self.class_names)
        img_inst_segs.instance_segmentation(self.model)
        num_inst       = img_inst_segs.segment['masks'].shape[2]
        num_inst_final = img_inst_segs.segment_['masks'].shape[2]
        issues         = img_inst_segs.issues
        return timepart, img_inst_segs, num_inst, num_inst_final, issues

    def inferencing_random_sample(self):
        file_name     = random.choice(self.list_f)
        timepart, img_inst_segs, num_inst, num_inst_final, issues = self.segmentation_inferencing(file_name)  
        print('There were {} instances segmented in {}, {} were left after quality control.\n'.format(num_inst, file_name, num_inst_final))
        self.result_visualization(img_inst_segs, updated=False, print_img=False, savename=None)
        if num_inst != num_inst_final:
            print('\nThe issues are: {}.\n'.format(issues))  
            self.result_visualization(img_inst_segs, updated=True, print_img=False, savename=None)

    def inferencing_all(self):
        print('start\n')
        print('...\n')
        count = 0
        csvfile = open(os.path.join(self.savedir, 'segmentation_summary.csv'), 'w', newline='')
        writer_junk = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_junk.writerow(['file_name', 'num_instances', 'type1_issue', 'type2_issue', 'num_instances_final'])
            
        for filename in self.list_f:
            timepart, img_inst_segs, num_inst, num_inst_final, issues = self.segmentation_inferencing(filename) 
            r  = img_inst_segs.segment
            r_ = img_inst_segs.segment_
            num_ins = r['masks'].shape[2]
            num_ins_final = r_['masks'].shape[2]
            writer_junk.writerow([timepart, num_inst, issues['issue1'], issues['issue2'], num_inst_final])
            count += 1
            self.result_visualization(img_inst_segs, updated=False, print_img=True, savename=timepart)
            self.result_visualization(img_inst_segs, updated=True, print_img=True, savename=timepart)
            if count == 1:
                print('1 image done. Which is {}\n'.format(count, filename))
            else:     
                print('{} images done. The last one is {}\n'.format(count, filename))
        csvfile.close()
        print('Done! \nThe results are saved here: {}. \nThe updated results are saved here: {}'.format(self.segmentation_dir, self.updated_dir))


