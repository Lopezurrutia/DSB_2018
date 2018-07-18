import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
import skimage.transform

import model as modellib
import pandas as pd

import os


import my_functions as f


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

## This is the path to your trained weights
model_path = os.path.join(MODEL_DIR,'YOUR_LOG_PATH','final.h5') 

## modify these paths to where stage_2 images are downloaded
sample_submission = pd.read_csv('stage2_sample_submission_final.csv')
test_path ='stage_2' 
    

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










###################################################################################################
## Function that merges the test time augmentation masks for each image
## Note, this is not really model ensemble but test time augmentation (TTA) merge
###################################################################################################
def ensemble_image(aug_masks,image_id,iou_threshold = 0.6,remove_duplicate_threshold=0.7,threshold_rem=4,file_normal='foo.csv'):
    """ Function to merge the TTA, takes a list of n_augs TTAs and returns a single mask
    # Arguments
       aug_masks is a list of numpy arrays each containing a TTA prediction
       image_id string with image id
       iou_threshold: when two masks from two TTAs have an iou higher that this threshold they are considered the same particle
       remove_duplicate_threshold: value passed to the remove_duplicate function. Remove duplicates basically removes those pixels belonging to two different mask, in addition when doing this, if a mask area is reduced by a percentage that is more than this threshold it is removed from the mask
       threshold_rem: when a particle has not been detected by at least this number of TTA it is removed from the mask
    # Returns:
       The image_id and encoded pixels for that image
    """



    ##############################################################################################
    ##### MERGE TWO MASKS 
    ##############################################################################################
    def merge_masks(mask_1,mask_2,iou_threshold = 0.6,remove_duplicate_threshold=0.7):
        """ Function to merge two masks
        Code is based on the iou code that was provided as a function to calculate mAP
        # Arguments
           mask_1 and mask_2: the two numpy arrays with the masks to be merged
           iou_threshold: when two masks have an iou higher that this threshold they are considered the same particle
        # Returns:
           A single mask with the merged masks
        """

        ## clip mask_1 to 0 1 so we can flatten by addition
        ## preserve mask_1 which can have values larger than 1
        mask_1_0 = np.clip(mask_1,0,1)
        mask_1_0 = f.remove_duplicate(mask_1_0,threshold=remove_duplicate_threshold)
        mask_1 = mask_1[:,:,np.sum(mask_1_0,axis = (0,1)) != 0]
        mask_1_0 = mask_1_0[:,:,np.sum(mask_1_0,axis = (0,1)) != 0]

        pred_labels_1 = mask_1_0 * (np.arange(mask_1_0.shape[2])+1) ## replace 1 by the number of mask
        pred_labels_1= np.sum(pred_labels_1,axis=2) ## flatten by addition

        ## flatten the second mask, this one has only 0 and 1s
        mask_2 = f.remove_duplicate(mask_2,threshold=remove_duplicate_threshold)
        pred_labels_2 = mask_2 * (np.arange(mask_2.shape[2])+1) ## replace 1 by the number of mask
        pred_labels_2= np.sum(pred_labels_2,axis=2) ## flatten by addition

        ## Number of objects for mask1 and mask2
        true_objects = len(np.unique(pred_labels_1))
        pred_objects = len(np.unique(pred_labels_2))

        # Compute intersection between all objects
        intersection = np.histogram2d(pred_labels_1.flatten(), pred_labels_2.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(pred_labels_1, bins = true_objects)[0]
        area_pred = np.histogram(pred_labels_2, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Compute the intersection over union
        iou = intersection / union
        best_match =(np.argmax(iou,axis=1)-1)[1:]
        best_match_iou =(np.max(iou,axis=1))[1:]
        best_match[best_match_iou<iou_threshold] = -1

        ## remove zeros as it is either background or no match
        second_labels= np.arange(pred_objects-1) ## background is not an object
        second_not_one = np.setdiff1d(second_labels,best_match)
        mask_4 = mask_2[:,:,second_not_one]  ## Masks in 2 not in 1, we divide by two

        ##  Append a zero mask at the end of  mask_2 that will be used for no matches
        zeromat= np.expand_dims(np.zeros_like(mask_2[:,:,0]),2)
        mask_3 = np.concatenate((mask_2,zeromat) ,axis = 2)
        mask_3 = mask_3[:,:,best_match] ## ordered mask_2 elements

        ## Compose the averaged masks
        merged_mask = np.sum(np.array([mask_1,mask_3]), axis=0)
        merged_mask = np.concatenate((merged_mask,mask_4) ,axis = 2)
        return merged_mask
    ##############################################################################################


    
    ## Default empty values in case no mask found
    ImageId_i =  [image_id]
    EncodedPixels_i = ['']
    
    mask_1 = aug_masks[0]
    for j in np.arange(1,len(aug_masks)):
        mask_2 = aug_masks[j]
        if (type(mask_1)== bool): ## Mask1 had no objects copy mask_2 over
            print('No valid mask_1, skip')
            mask_1 = mask_2
        else:
            if (type(mask_2)!= bool): ## Both masks have objects
                print('Merging',j,' for image' , image_id)
                mask_1 = merge_masks(mask_1,mask_2,iou_threshold = 0.6,remove_duplicate_threshold= remove_duplicate_threshold)
            else:
                print('No Merge, keep mask_1 as is')
    if (type(mask_1)!= bool): # otherwise after merging we still have no particles detected
        ## Remove pixels that were not detected by at least threshold_rem TTA and empty masks
        mask_1 = (mask_1>threshold_rem).astype('uint8') 
        _idx = np.sum(mask_1, axis=(0, 1)) > 0
        mask_1 = mask_1[:, :, _idx]


        if(mask_1.shape[2]>0):  ## Image has no detected nuclei
            ImageId_batch, EncodedPixels_batch, mask_1 = f.numpy2encoding(mask_1, image_id)
            ImageId_i = ImageId_batch
            EncodedPixels_i = EncodedPixels_batch

            ImageId_batch, EncodedPixels_batch, mask_1 = f.numpy2encoding(mask_1, image_id,dilation=True)
    lock.acquire()
    df = pd.DataFrame({ 'ImageId' : ImageId_i , 'EncodedPixels' : EncodedPixels_i})
    df.to_csv(file_normal, index=False,mode='a', columns=['ImageId', 'EncodedPixels'],header= False)

    lock.release()
    
    return 0
###################################################################################################





###################################################################################################
## Define a single thread queue
## that will be merging the image masks while the TTA continues on other images
###################################################################################################

def worker(queue) :
    while True :
        items = queue.get(True)
        ensemble_image(items[0],items[1],iou_threshold = items[2],remove_duplicate_threshold=items[3],threshold_rem=items[4],file_normal=items[5])

import multiprocessing
lock = multiprocessing.Lock()
###################################################################################################




# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


## We will predict on 15 different augmentations:
n_augs = 15
# FlipUD || FlipLR || ROT90  || CHANNEL_SHIFT_RANGE || ENLARGE_W || ENLARGE_H
aug_options = [[False, False, 0, 0,1,1],
               [False, False, 0, 0,2,2],
               [False, False, 0, 0,0.7,0.8],
               [False, False, 0, 15,1.5,1.5],
               [True, True, 1,0,0.9,0.8],
               [True, False, 2,7,1.1,1.2],
               [False,True, 3,10,0.5,0.5],
               [False,True, 3,10,1,1.2],
               [True,True, 2,25,1,1],
               [True,True, 2,25,1,1],
               [False,True, 3,5,1,1],
               [True,False, 1,15,1,1],
               [True,True, 3,15,1,0.9],
               [False,True, 3,10,1.5,1.5],
               [False, False, 3,5,1.2,1]]




ImageId = []
EncodedPixels = []

start=time.time()

iou_threshold = 0.6
remove_duplicate_threshold=0.7
threshold_rem=4

file_normal='submission_tta.csv'
## Write the headers of the submission files
df = pd.DataFrame({ 'ImageId' : [] , 'EncodedPixels' : []})
df.to_csv(file_normal, index=False, columns=['ImageId', 'EncodedPixels'])



## Multiproccesing job list
jobs=[]

n_images= len(sample_submission.ImageId)
for i in np.arange(n_images):
    image_id = sample_submission.ImageId[i]
    ##Set seeds for each image
    print('Start detect',i, '  ' ,image_id)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

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

    bzResults = [] 
    ## Predict for each augmentation
    for j in np.arange(n_augs): ## for each augmentation
        print('Model', j, 'Image',i, 'id ', image_id)
        flipud, fliplr, nrot, shift_range, h_ratio, w_ratio = aug_options[j]
        ## IMAGE AS DIRECTLY READ FROM DISK
        image = np.copy(original_image)
        if flipud:
            image = np.flipud(image)
        if fliplr:
            image = np.fliplr(image)
        if nrot:
            image = np.rot90(image, k=nrot,axes=(0,1))
        # if shift_range:
        #     image = random_channel_shift(image,shift_range,2)
        h, w = image.shape[:2] ## has to be after rotations
        if h_ratio !=1 or w_ratio!= 1:
            if ((h <512 or w < 512) and (h_ratio<1 or w_ratio<1)):##
                print('skip reduce, image too small',image.shape,h_ratio,w_ratio)
            else:
                scale = (h_ratio, w_ratio) ## change aspect ratio and enlarge
                round_scale = (round(h * scale[0]), round(w * scale[1]))
                image = skimage.transform.resize(
                    image, (round(h * scale[0]), round(w * scale[1])),
                    order=1, mode="constant", preserve_range=True)


        ## Make a prediction for this image and augmentation
        results = model.detect([image], verbose=0)
        r=results[0]
        mask = results[0]['masks']
        print('End detection')

        ## If no particle detected the code returns masks as minimasks
        ## convert to False
        if len(r['class_ids']):

            ## Now we have to reverse the image augmentations so we get back the original image
            h_2,w_2 = mask.shape[:2] ## has to be before rotations, reverse order augmentation
            if h_ratio !=1 or w_ratio!= 1:
                if ((h <512 or w < 512) and (h_ratio<1 or w_ratio<1)):##
                    print('skip reduce, image too small',image.shape,h_ratio,w_ratio)
                else:
                    mask = scipy.ndimage.zoom(mask, zoom=[h/h_2, w/w_2,  1], order=0)
            if nrot:
                mask = np.rot90(mask, k=4-nrot,axes=(0,1))
            if fliplr:
                mask = np.fliplr(mask)
            if flipud:
                mask = np.flipud(mask)
            if not np.all(mask.shape[:2] == original_image.shape[:2]):
                print ('Mask shape', mask.shape, 'and image shapes', original_image.shape, ' differ')
                break

        else:
            mask = False

        print('End ')

        bzResults.append(mask) ## append mask to the list of augmented masks for that image

    p= multiprocessing.Process(target=ensemble_image,args=(bzResults,image_id,iou_threshold,remove_duplicate_threshold,threshold_rem,file_normal,))
    jobs.append(p)
    p.start()

    #######################################################################
    ## Do not overpopulate the multiproccessing jobs
    MAXPROCCESES = 10
    wait_for_proccesses=True
    while wait_for_proccesses:
        for proc_i in np.flip(np.arange(len(jobs)),0): ## Need reverse ordering so we can delete without changing indexing
            proc= jobs[proc_i]
            proc.join(timeout=0)
            if not proc.is_alive(): ## Proccess has finished, will remove it from list
                del jobs[proc_i]
            
        if len(jobs)<MAXPROCCESES: 
            wait_for_proccesses = False
        print("Alive Jobs",len(jobs))


## Wait for all proccesses to finish
for j in jobs:
    j.join()
    
end = time.time()

