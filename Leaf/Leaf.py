#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:13:51 2019

@author: wzhan
"""

"""
Mask R-CNN
Train on the Synthetic Arabidopsis dataset which based on Leaf Challenging Segmentation 
https://data.csiro.au/collections/#collection/CIcsiro:34323v004. 
Download the dataset and put it under the Mask_RCNN directory 

Written by Noah Falhgren and Wenxiao Zhan

------------------------------------------------------------
"""

# Set matplotlib backend 
# This has to be done before other import that might set it
# But only if we're running in script mode. 

if __name__ == '__main__':
    import matplotlib
    # Set 'Agg' as backend which cant display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
# %% importing module 
import os 
import sys
import datetime 
import numpy as np 
# To import gen_IDs.py, add the root directory of gen_IDs to a python 
# package searching file. You can modify this according to your path
sys.path.append('')
import gen_IDs
from imgaug import augmenters as iaa 
from plantcv import plantcv as pcv 


# Root directory of the project. You can modify according to your path 
ROOT_DIR = '/mnt/efs/data/Mask_RCNN'

# Import Mask RCNN 
sys.path.append(ROOT_DIR)
from mrcnn.config import Config 
from mrcnn import utils 
from mrcnn import model as modellib 
from mrcnn import visualize 

# %% Preparation of dataset, project directory.. 

# Path to trained weights file. Put the pre-trained weights file under ROOT_DIR 
LEAF_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_leaf.h5')

# Direcotry of dataset. Modify it according to your path 
dataset_dir ='/mnt/efs/data/synthetic_arabidopsis_LSC'

# Generate the list of image_id for train, validation and test dataset 
# The generate_ID function assume the image and mask are stored like 
# synthetic_arabidopsis (image file and mask file are under dataset_dir. 
# There is no sub folder here)
# If you have different structure of dataset, please modify the generate_IDs
# function in the gen_IDs.py
num_images, Image_IDs_train, Image_IDs_val, Image_IDs_test = gen_IDs.generate_IDs(dataset_dir)

# Direcotry to save logs and model checkpoints, if not provided 
# through the command line argument --logs 
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Results directory 
RESULTS_DIR = os.path.join(ROOT_DIR, 'results/leaves')


# %% Set Hyperparameter for Training 

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


# %% Set Hyperparameter for Testing 
    
class LeavesInferenceConfig(LeavesConfig):
    # Set batch size to 1 to run and inference one image at a time 
    GPU_COUNT = 1 
    IMAGES_PER_GPU =1 
    # Don't resize image for inferencing 
    IMAGE_RESIZE_MODE = 'pad64'
    # Non-max suppression threhold to filter RPN proposals 
    # You can increase this during training to generate more proposals
    RPN_NMS_THRESHOLD = 0.9 
    

# %% Load Dataset image and mask 
    
class LeavesDataset(utils.Dataset):
    """Load the synthetic arabidopsis dataset. 
    Different image from dataset has different name. In this load_leaves function, we assume 
    image name is like format of 'plant00000_rgb.png' and corresponding mask is like 
    'plant00000_label.png'. The image_id generated by function gen_IDs is like 'plant00000'. 
    """
    
    def load_leaves(self, dataset_dir, Image_IDs):
        """Load a subset of the leaf dataset. 
        
        dataset_dir: Root direcotry of the dataset 
        Image_IDs: A list that includes all the images to load 
        """
        # Add classes. We only have one class to add---leaves 
        # Naming the dataset 'leaves'
        self.add_class('leaves', 1, 'leaves')
        
        # Add images 
        for image_id in Image_IDs:
            self.add_image(
                    'leaves',
                    image_id = image_id,
                    path = os.path.join(dataset_dir, image_id + '_rgb.png')) ##

        
    def load_mask(self, image_id):
        '''
        Generate masks for each instance of the given image 
        Input: the image_id, the index number of given image in the Image_IDs_(train, val, test)
        Output: 
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance. 
        class_ids: a 1D array of class IDs of the instance masks. Here we only have 
        one class, leave and class_id = 1. 
        '''
        info = self.image_info[image_id]
        mask_id = '{}_label.png'.format(info['id'])
        
        instance_masks = []
        
        # Set debug to 'print' instead of None
        pcv.params.debug = 'print'
        
        # Use a pre-v3 function to open a image 
        # Note that pcv.params.debug is passed to the debug argument
        # read the mask file for the given image
        masks_img, masks_path, masks_filename = pcv.readimage(os.path.join(dataset_dir, mask_id))  ## path? 
        # Find the pixel value for each mask
        uniq_rgb = np.unique(masks_img.reshape(-1, masks_img.shape[2]), axis=0) 
        ## remove the instance of background, whose pixel value is [0, 0, 0]
        uniq_rgb = np.delete(uniq_rgb, 0, 0) 
        for rgb in uniq_rgb:
            # Generate a mask for each instance with positive area = 255 
            # negative area = 0 
            mask = np.zeros(np.shape(masks_img)[:2], dtype=np.uint8)
            mask[np.all(masks_img == rgb, axis=-1)] = 255
            # Transfer the mask to bool mask. Positive area = True 
            # Negative area = False
            mask = mask.astype(np.bool)
            instance_masks.append(mask)
        instance_masks = np.stack(instance_masks, axis=-1)
        return instance_masks, np.ones(uniq_rgb.shape[0], dtype=np.int32)
        
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info['source'] == 'leaves':
            return info['id']
        else:
            super(self.__class__, self).image_reference(image_id)
    
# %% Training 
            
def train(model, dataset_dir):
    """ Train the model. """
    
    # Preparing training and validation dataset
    ## Generate LeavesDataset class. 
    dataset_train = LeavesDataset()
    dataset_val = LeavesDataset()
    
    ## Preparing training dataset.
    dataset_train.load_leaves(dataset_dir, Image_IDs_train)
    dataset_train.prepare()
    
    ## Preparing validation dataset 
    dataset_val.load_leaves(dataset_dir, Image_IDs_val)
    dataset_val.prepare()
        
    
    # Image augmentation 
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from mask_rcnn_leave.h5, train heads only for a bit
    # since they have random weights
    print("Train network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs = 20, # Discuss with Dr Noah 
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs = 60, # Discuss with Dr Noah 
                augmentation=augmentation,
                layers='all')
    
# %%RLE Encoding 
    
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)



# %% Inferencing 

def detect(model, dataset_dir):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory to store detecting result. 
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Load test dataset
    dataset_test = LeavesDataset()
    dataset_test.load_leaves(dataset_dir, Image_IDs_test)
    dataset_test.prepare()
    # Load over images
    submission = []
    for image_id in dataset_test.image_ids:
        # Load image and run detection
        image = dataset_test.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset_test.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset_test.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset_test.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


##%% Running on spyder directory instead command line
#
## Training 
### Configurations
#config = LeavesConfig()
##config.display()
### Create model 
#print('in mask RCNN +++++++++++++++++++++++++++++++++++++++++++++++')
#model = modellib.MaskRCNN(mode='training', config=config, model_dir= DEFAULT_LOGS_DIR)
### Select weights file to load
#weights_path = LEAF_WEIGHTS_PATH
#model.load_weights(weights_path, by_name=True)
#train(model, dataset_dir)

# %% Running using Command Line parsing  

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for leaf counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset_dir', required=False,
                        metavar="/path/to/dataset_dir/",
                        help='Root directory of the dataset_dir')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset_dir, "Argument --dataset_dir is required for training"
    elif args.command == "detect":
        assert args.dataset_dir, "Provide --dataset_dir to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset_dir)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LeavesConfig()
    else:
        config = LeavesInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        print('in mask RCNN +++++++++++++++++++++++++++++++++++++++++++++++')
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset_dir)
    elif args.command == "detect":
        detect(model, args.dataset_dir)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

