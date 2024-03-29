"""
Modified from:

Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


# Modified version of display_instances - saves image to specified folder

def save_instances(image, boxes, masks, class_ids, class_names, gt_mask,
					  scores=None, title="",
					  figsize=(16, 16), ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None, save_dir = None, save_name = None,
						gt_matches = [], pred_matches = [], label = False):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""
	# Number of instances
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	# If no axis is passed, create one and automatically call show()
	auto_show = False
	if not ax:
		_, ax = plt.subplots(1, figsize=figsize)
		auto_show = True

	# Generate random colors
	colors = colors or random_colors(N)

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	ax.set_ylim(height + 10, -10)
	ax.set_xlim(-10, width + 10)
	ax.axis('off')
	ax.set_title(title)

	masked_image = image.astype(np.uint32).copy()
	
	#  Ground truth Mask
	N_gt = gt_mask.shape[2]
	
	if N_gt != len(gt_matches):
		print("GT mask shape = " + str(N_gt) + ", GT matches shape = " + str(len(gt_matches)))
	
	for i in range(N_gt):
		mask = gt_mask[:, :, i]
		#if show_mask:
			#masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		

		# Set colour for ground truth instances:
		if i <= (len(gt_matches)-1):		

			if gt_matches[i] != -1:
				edgecolor = 'white'
				linestyle = 'solid'
			else:
				edgecolor = 'black' # False negative
				linestyle = 'dotted'

			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				p = Polygon(verts, facecolor="none", edgecolor=edgecolor, linestyle =linestyle, linewidth = 5)
				ax.add_patch(p)

	# Prediction masks:
	for i in range(N):
		color = colors[i]

		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
								alpha=0.7, linestyle="dashed",
								edgecolor=color, facecolor='none')
			ax.add_patch(p)

		# Label
		if label == True:		

			if not captions:
				class_id = class_ids[i]
				score = scores[i] if scores is not None else None
				label = class_names[class_id]
				x = random.randint(x1, (x1 + x2) // 2)
				caption = "{} {:.3f}".format(label, score) if score else label
			else:
				caption = captions[i]
			ax.text(x1, y1 + 8, caption,
					color='w', size=11, backgroundcolor="none")

		# Mask
		mask = masks[:, :, i]
		#if show_mask:
			#masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		
		if pred_matches[i] != -1:
			edgecolor = (1,0,1) # True positive
			linestyle = 'solid'
		else:
			edgecolor = (1,0,1) # False positive
			linestyle = 'dashed'

		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=edgecolor, linestyle = linestyle, linewidth = 5)
			ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	if auto_show:
		#plt.show()
		plt.savefig(save_dir + save_name)

## Un modified: ###

def random_colors(N, bright=True):
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

def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + alpha * color[c] * 255,
								  image[:, :, c])
	return image

