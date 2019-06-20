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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib
import mrcnn.utils as utils
import mrcnn.visualize
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

image_dir = "" # Path to directory containing training, validation and test images.
gt_dir = "" # Path to dir of .npy ground truth masks.

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_WEIGHTS_PATH):
	utils.download_trained_weights(COCO_WEIGHTS_PATH)
	
print("Coco dir = "+COCO_WEIGHTS_PATH)


class nlbConfig(Config):
	"""Configuration for training on 512x512 pix nlb images.
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




# ## Dataset
# 
# 
# Extend the Dataset class and add a method to load the masks.
# 
# * load_image()
# * load_mask()
# * image_reference()

class nlbDataset(utils.Dataset):
	
	
	# Adds 'name' and 'augment' fields to add_image class.
	def add_image(self, source, image_id, path, name, augment, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
			"name": name,
			"augment": augment,	
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)
		
		return image_info
	
	## Augmentation codes:

	# 0 = none
	# 1 = fliplr
	# 2 = rotate 90
	# 3 = rotate 90 + fliplr
	# 4 = rotate 180
	# 5 = rotate 180 + fliplr
	# 6 = rotate 270
	# 7 = rotate 270 + fliplr


	# Adds image augmentation step to original function:
	def load_image(self, image_id):
		"""
		Load the specified image and return a [H,W,3] Numpy array.
		"""
		# Load image
		image = skimage.io.imread(self.image_info[image_id]['path'])
		
		# If grayscale. Convert to RGB for consistency.
		if image.ndim != 3:
			image = skimage.color.gray2rgb(image)
		
		# Augmentation:
		
		# Flip:
		if self.image_info[image_id]['augment'] == 1:
			image = np.fliplr(image)
		# Rotate 90
		if self.image_info[image_id]['augment'] == 2:
			image = np.rot90(image, 1)
		# Rotate 90 + flip
		if self.image_info[image_id]['augment'] == 3:
			image = np.fliplr(np.rot90(image, 1))
		# Rotate 180
		if self.image_info[image_id]['augment'] == 4:
			image = np.rot90(image, 2)
		# Rotate 180 + flip
		if self.image_info[image_id]['augment'] == 5:
			image = np.fliplr(np.rot90(image, 2))
		# Rotate 270
		if self.image_info[image_id]['augment'] == 6:
			image = np.rot90(image, 3)
		# Rotate 270 + flip
		if self.image_info[image_id]['augment'] == 7:
			image = np.fliplr(np.rot90(image, 3))

		return image
	

	def load_mask(self, image_id):
		
		"""
		Loads masks for each image
		"""
				  
		image_name=self.image_info[image_id]['name'] # get image name
		image_file=image_name+".jpg" # Add file extension to name	   

		mask_dir= gt_dir # dir of image ground truth masks
		
		mask_file = mask_dir+image_name+"_mask.npy"

		mask = np.load(mask_file)

		# Augmentation:
		
		# Flip:
		if self.image_info[image_id]['augment'] == 1:
			mask = np.fliplr(mask)
		# Rotate 90
		if self.image_info[image_id]['augment'] == 2:
			mask = np.rot90(mask, 1)
		# Rotate 90 + flip
		if self.image_info[image_id]['augment'] == 3:
			mask = np.fliplr(np.rot90(mask, 1))
		# Rotate 180
		if self.image_info[image_id]['augment'] == 4:
			mask = np.rot90(mask, 2)
		# Rotate 180 + flip
		if self.image_info[image_id]['augment'] == 5:
			mask = np.fliplr(np.rot90(mask, 2))
		# Rotate 270
		if self.image_info[image_id]['augment'] == 6:
			mask = np.rot90(mask, 3)
		# Rotate 270 + flip
		if self.image_info[image_id]['augment'] == 7:
			mask = np.fliplr(np.rot90(mask, 3))



		## Create 1D array of class IDs	
		
		# Handle occurences where mask only has 1 slice:
		if len(mask.shape) == 3:
			class_ids=[1]*mask.shape[2]
		else:
			class_ids=[1]
			mask.shape += 1, # Change shape from [n,m] to [n,m,1]
		#class_ids=class_ids.astype(np.int32)
		class_ids = np.array(class_ids)
			
		return mask, class_ids
	
	

## Training dataset ##

dataset_train = nlbDataset()

## Add instance classes:

dataset_train.add_class("nlb", 1, "lesion")


print('classes added')


## Get list of training images:

imageDir= image_dir + 'train/'
imageList=os.listdir(imageDir)


## Add images

for i in enumerate(imageList):
	
	# Add augmentation code:
	for augment in range(8):

		dataset_train.add_image("nlb", i[0], imageDir+imageList[i[0]], os.path.splitext(imageList[i[0]])[0], augment)
	

dataset_train.prepare()



## Validation dataset ##

dataset_val = nlbDataset()

## Add instance classes:


dataset_val.add_class("nlb", 1, "lesion")


print('classes added')


## Get list of validation images:
imageDir= image_dir +'val_img/'
imageList=os.listdir(imageDir)


## Add images

for i in enumerate(imageList):
	# Add augment code for each image:
	for augment in range(8):

		dataset_val.add_image("nlb", i[0], imageDir+imageList[i[0]], os.path.splitext(imageList[i[0]])[0], augment)

dataset_val.prepare()



## Ceate Model ##

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)


# Which weights to start with?

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
	model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
	# Load weights trained on MS COCO, but skip layers that
	# are different due to the different number of classes
	# See README for instructions to download the COCO weights
	model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
	# Load the last model you trained and continue training
	model.load_weights(model.find_last()[1], by_name=True)


# ## Training


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=4, layers='heads')


# Train all layers
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=5, layers='all')

# Use smaller learning rate to fine tune all layers:
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/100, epochs=10, layers='all')


