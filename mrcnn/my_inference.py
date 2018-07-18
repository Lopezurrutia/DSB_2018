seed=123
from keras import backend as K

import numpy as np
np.random.seed(seed)
import tensorflow as tf

tf.set_random_seed(seed)

import random
random.seed(seed)

import skimage.io 
from skimage import img_as_ubyte

import model as modellib
import pandas as pd
import os

import my_functions as f


#######################################################################################
## SET UP CONFIGURATION
from config import Config

class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"

    IMAGE_RESIZE_MODE = "pad64" ## tried to modfied but I am using other git clone
    ## No augmentation
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False ## Not using this

    IMAGE_MIN_DIM = 512 # We scale small images up so that smallest side is 512
    IMAGE_MAX_DIM = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD =  0.2
    DETECTION_MIN_CONFIDENCE = 0.9

    LEARNING_RATE = 0.001
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    USE_MINI_MASK = True


inference_config = BowlConfig()
inference_config.display()
#######################################################################################


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


## Change this with the path to the last epoch of train
model_path = 
model_path = os.path.join(MODEL_DIR,'YOUR_LOG_FOLDER','final.h5')


## change this with the correct paths for images and sample submission
test_path = os.path.join(ROOT_DIR,'stage_2')
sample_submission = pd.read_csv('stage2_sample_submission_final.csv')


print("Loading weights from ", model_path)


import time
start_time = time.time()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
model.load_weights(model_path, by_name=True)


ImageId_d = []
EncodedPixels_d = []

n_images= len(sample_submission.ImageId)
for i in np.arange(n_images):
    image_id = sample_submission.ImageId[i]
    print('Start detect',i, '  ' ,image_id)
    ##Set seeds for each image, just in case..
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ## Load the image
    image_path = os.path.join(test_path, image_id, 'images', image_id + '.png')
    original_image = skimage.io.imread(image_path)
    ####################################################################
    ## This is needed for the stage 2 image that has only one channel
    if len(original_image.shape)<3:
        original_image = img_as_ubyte(original_image)
        original_image = np.expand_dims(original_image,2)
        original_image = original_image[:,:,[0,0,0]] # flip r and b
    ####################################################################
    original_image = original_image[:,:,:3]

    ## Make prediction for that image
    results = model.detect([original_image], verbose=0)

    ## Proccess prediction into rle
    pred_masks = results[0]['masks']
    scores_masks = results[0]['scores']
    class_ids = results[0]['class_ids']

    if len(class_ids): ## Some objects are detected
        ImageId_batch, EncodedPixels_batch, _ = f.numpy2encoding(pred_masks, image_id,scores=scores_masks,dilation=True)
        ImageId_d += ImageId_batch
        EncodedPixels_d += EncodedPixels_batch

    else:
        print('No particles detected',i,pred_masks.shape)
        ImageId_d +=  [image_id]
        EncodedPixels_d += ['']


f.write2csv('submission.csv', ImageId_d, EncodedPixels_d)

end_time = time.time()
ellapsed_time = (end_time-start_time)/3600
print('Time required to train ', ellapsed_time, 'hours')

