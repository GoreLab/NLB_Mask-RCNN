import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from skimage import io
import skimage
import glob
import fnmatch
import tensorflow as tf
import csv
import pandas as pd

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib
import mrcnn.utils as utils
import mrcnn.visualize
from mrcnn.model import log

#### SETINGS ###

# Get all log files:
log_dir = '' # Path to training logs

# Epoch to start from:
start_epoch = 0


# Create output file:
out_file_name = 'out_file.csv' # Name of output file

headers=['log','epoch', 'mean_AP', 'mean_IoU', 'mean_precisions', 'mean_recalls','gt_instances','pred_instances','gt-pred']

if not os.path.exists(out_file_name):
	with open(out_file_name, 'w') as out_file:
		iou_writer = csv.writer(out_file)
		iou_writer.writerow(headers)
else:
	print('out file exists')
		
#### END SETTINGS ####



# Root directory of the project
ROOT_DIR = os.getcwd()

print("Root dir = "+ROOT_DIR)


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

image_dir = '' # Directory of test images
gt_dir = '' # Directory of ground trugth masks

DEVICE = "/gpu:1"

## Configurations ##


## Configurations ##
class nlbConfig(Config):
	"""
	Configuration for 512x512 pix nlb images.
	Derives from the base Config class and overrides values specific
	to the dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "nlbTrain"

	# Train on 2 GPU and 4 images per GPU. 
	GPU_COUNT = 2
	IMAGES_PER_GPU = 4

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # background + lesion.

	# Default = 1000
	STEPS_PER_EPOCH = 1000

	# Default = 50
	VALIDATION_STEPS = 50

	# Don't exclude based on confidence. Since we have two classes
	# then 0.5 is the minimum anyway as it picks between lesion and BG
	DETECTION_MIN_CONFIDENCE = 0

	# Backbone network architecture
	# Supported values are: resnet50, resnet101
	BACKBONE = "resnet101"

	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	
	IMAGE_RESIZE_MODE = "none" # Images are already correct size
	IMAGE_MIN_DIM = 512
	IMAGE_MAX_DIM = 512

	# Use smaller anchors because our image and objects are small
	#RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

	# ROIs kept after non-maximum supression (training and inference)
	POST_NMS_ROIS_TRAINING = 1000
	POST_NMS_ROIS_INFERENCE = 2000

	# Non-max suppression threshold to filter RPN proposals.
	# You can increase this during training to generate more propsals.
	RPN_NMS_THRESHOLD = 0.9

	# How many anchors per image to use for RPN training
	RPN_TRAIN_ANCHORS_PER_IMAGE = 64

	# Image mean (RGB)
	MEAN_PIXEL = np.array([123.64, 138.04, 91.0])
	#mean R = 123.64074478785197
	#mean G = 138.0523718401591
	#mean B = 91.00018168258667

	# If enabled, resizes instance masks to a smaller size to reduce
	# memory load. Recommended when using high-resolution images.
	USE_MINI_MASK = True
	MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

	# Number of ROIs per image to feed to classifier/mask heads
	# The Mask RCNN paper uses 512 but often the RPN doesn't generate
	# enough positive proposals to fill this and keep a positive:negative
	# ratio of 1:3. You can increase the number of proposals by adjusting
	# the RPN NMS threshold.
	TRAIN_ROIS_PER_IMAGE = 128

	# Maximum number of ground truth instances to use in one image
	MAX_GT_INSTANCES = 200

	# Max number of final detections per image
	DETECTION_MAX_INSTANCES = 400
	
	# Default = 0.001
	LEARNING_RATE = 0.001

	
config = nlbConfig()
config.display()


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(nlbConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
inference_config.display()





# ## Dataset
# 
# 
# Extend the Dataset class and add a method to load the masks.
# 
# * load_image()
# * load_mask()
# * image_reference()

class nlb500Dataset(utils.Dataset):
	
	
	# Adds 'name' to add_image class.
	def add_image(self, source, image_id, path, name, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
			"name": name,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)
		
		return image_info
	
	
	def load_mask(self, image_id):
		
		"""
		Loads masks for each image
		"""
				  
		image_name=self.image_info[image_id]['name'] # get image name
		image_file=image_name+".jpg" # Add file extension to name	   
		mask_dir= gt_dir # dir of image ground truth masks
		mask_file = mask_dir+image_name+"_mask.npy"
		
		mask = np.load(mask_file)


		## Create 1D array of class IDs	
		class_ids=[1]*mask.shape[2]
		class_ids = np.array(class_ids)
			
		return mask, class_ids
	
	




## Mini-Validation dataset ##

dataset_minival = nlbDataset()

## Add instance classes:


dataset_minival.add_class("nlb", 1, "lesion")


print('classes added')


## Get list of test images:
imageDir= image_dir + 'test_img/'

imageList=os.listdir(imageDir)

## Add images

for i in enumerate(imageList):
   
	dataset_minival.add_image("nlb", i[0], imageDir+imageList[i[0]], os.path.splitext(imageList[i[0]])[0])


dataset_minival.prepare()



## Ceate Model ##



## Detection


print("model_dir = "+DEFAULT_LOGS_DIR)

# Recreate the model in inference mode
with tf.device(DEVICE):
	model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=DEFAULT_LOGS_DIR)


# Get all logs:
all_logs = [x for x in os.listdir(log_dir) if x.endswith(".h5")]

# Get logs greater than start epoch:

logs=[]

for i in all_logs:
	if int(i.split('_')[3].split('.')[0]) > start_epoch:
		logs.append(i)

logs.sort()



## Calculate AP and IoU for each epoch:

mage_ids = dataset_minival.image_ids

for log in logs:

	# Get path to saved weights
	model_path = os.path.join(ROOT_DIR, log_dir,log)

	# Load trained weights (fill in path to trained weights here)
	assert model_path != "", "Provide path to trained weights"
	print("Loading weights from ", model_path)
	model.load_weights(model_path, by_name=True)

	
	APs = []
	IoUs = []
	precisions_out = []
	recalls_out = []
	gt_instances = 0
	pred_instances =0

		

	for image_id in image_ids:
		# Load image and ground truth data
		image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_minival, inference_config, image_id, use_mini_mask=False)
		
		
		# Run object detection
		results = model.detect([image], verbose=0)
		r = results[0]
		
		
		# Compute AP
		AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r["masks"])
		APs.append(AP)
	
		

		## Calculate IoU

		# Flatten pred masks
		pred_masks = results[0]['masks']
			
		if pred_masks.shape[0] == 512:
			pred_mask_sum = np.sum(pred_masks, axis=-1)
			pred_mask_bool = pred_mask_sum.clip(max=1)
		
			# Flatten GT masks
			gt_mask_sum = np.sum(gt_mask, axis=-1)
			gt_mask_bool=gt_mask_sum.clip(max=1)

			# Calculate intersect and union:
			iou_mask = pred_mask_bool+gt_mask_bool
		
			intercept = (iou_mask == 2).sum()
			union = (iou_mask != 0).sum()
		
			iou = intercept/union
			pred_instances += pred_masks.shape[2]
		else:
			iou = 0
			print("iou else loop")


		#print("IoU = "+str(iou))
		IoUs.append(iou)
		precisions_out.append(np.mean(precisions))
		recalls_out.append(np.mean(recalls))
		gt_instances += gt_mask.shape[2]
	


	# Split log file name to get epoch:		
	epoch_num = log.split('_')[3].split('.')[0]
	
	# Write output:
	with open(out_file_name, mode='a') as out_file:
		iou_writer = csv.writer(out_file)		
		iou_writer.writerow([log, epoch_num, np.mean(APs), np.mean(IoUs), np.mean(precisions_out), np.mean(recalls_out), gt_instances, pred_instances, gt_instances-pred_instances])
	
	
	print('mean AP = '+str(np.mean(APs))+'mean IoU = '+str(np.mean(IoUs)))


