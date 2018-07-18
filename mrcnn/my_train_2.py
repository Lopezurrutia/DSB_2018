seed=123

import numpy as np
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

import random
random.seed(seed)

import os

import sys

import time

from my_bowl_dataset import BowlDataset

import model as modellib
from model import log

import numpy as np
from imgaug import augmenters as iaa

import skimage.io 



#######################################################################################
## SET UP CONFIGURATION
from config import Config

class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "002"

    IMAGE_RESIZE_MODE = "crop"
    # Augmentation parameters
    ASPECT_RATIO = 1.3 ## Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1.2 ## Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.5 ## Maximum zoom per image, note that this will be randomized
    IMAGE_MIN_SCALE = False ## Not using this

    ROT_RANGE = 10.
    CHANNEL_SHIFT_RANGE = 15
    
    LEARNING_RATE = 0.001
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (4, 8 , 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = 664//IMAGES_PER_GPU
    VALIDATION_STEPS = 2//IMAGES_PER_GPU ## We are training with the whole dataset so validation is not very meaningfull, I put a two here so it is faster. We either use train loss or calculate in a separate procceses the mAP for each epoch

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512

bowl_config = BowlConfig()
bowl_config.display()
#######################################################################################

# Root directory of the project
ROOT_DIR = os.getcwd()


## Change this dir to the stage 1 training data
train_dir = os.path.join(ROOT_DIR,'stage1_train')


# Get train IDs
train_ids = next(os.walk(train_dir))[1]

## Get an idea of img_sizes
train_paths = [os.path.join(train_dir, train_id,'images','{}.png'.format(train_id)) for train_id in train_ids]
# print(train_paths)
img_sizes=[]
for i in train_paths:
    image = skimage.io.imread(i)
    img_sizes.append(image.shape)

set(img_sizes)

## First pass to count instances
## same as:
# from collections import Counter
# Counter(img_sizes)

counts = []
for i in set(img_sizes):
    
    count = 0
    for j in np.arange(len(img_sizes)):
        if img_sizes[j] == i:
            # train_ids_size
            count = count +1
    print(i,count)
    counts.append(count)


long_train_ids = []
replic_n = max(counts)
for i in set(img_sizes):
    train_ids_size = []
    count = 0
    for j in np.arange(len(img_sizes)):
        if img_sizes[j] == i:
            train_ids_size.append(train_ids[j])
            # train_ids_size
            count = count +1
    repeticiones = int(replic_n/count)
    print(i,count,repeticiones)
    train_ids_size = np.repeat(train_ids_size,repeticiones)
    long_train_ids.extend(train_ids_size.tolist())




train_ids = [os.path.join(train_dir, train_id) for train_id in long_train_ids]

# Training dataset
dataset_train = BowlDataset()
dataset_train.load_bowl(train_ids)
dataset_train.prepare()

# # Validation dataset, same as training.. will use pad64 on this one
dataset_val = BowlDataset()
dataset_val.load_bowl(train_ids)
dataset_val.prepare()






# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

model = modellib.MaskRCNN(mode="training", config=bowl_config,
                          model_dir=MODEL_DIR)

## Change this with the path to the last epoch of train first step
model_path = os.path.join(MODEL_DIR,'YOUR_001_DIR','final.h5') 
model.load_weights(model_path, by_name=True)

import time
start_time = time.time()

## Augment True will perform flipud fliplr and 90 degree rotations on the 512x512 images

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html

# ## This should be the equivalente version of my augmentations using the imgaug library
# ## However, there are subtle differences so I keep my implementation
# augmentation = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
#     iaa.OneOf([iaa.Affine(rotate=0),
#                iaa.Affine(rotate=90),
#                iaa.Affine(rotate=180),
#                iaa.Affine(rotate=270)]),
#     iaa.Sometimes(0.5,iaa.Affine(rotate=(-10,10))),
#     iaa.Add((-15, 15), per_channel=1)
    
# ])

augmentation=False

model.train(dataset_train, dataset_val, 
            learning_rate=bowl_config.LEARNING_RATE/30,
            epochs=10,
            augmentation=augmentation,
            layers="all")

model.train(dataset_train, dataset_val, 
            learning_rate=bowl_config.LEARNING_RATE/100,
            augmentation=augmentation,
            epochs=40, 
            layers="all")


end_time = time.time()
ellapsed_time = (end_time-start_time)/3600

print(model.log_dir)
model_path = os.path.join(model.log_dir, 'final.h5')
model.keras_model.save_weights(model_path)

