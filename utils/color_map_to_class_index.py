import cv2
import sys
import os.path
import numpy as np

def apply_potsdam_colormap(image):
	# Potsdam colormap
	# Impervious surfaces (RGB: 255, 255, 255)
	# Building (RGB: 0, 0, 255)
	# Low vegetation (RGB: 0, 255, 255)
	# Tree (RGB: 0, 255, 0)
	# Car (RGB: 255, 255, 0)
	# Clutter/background (RGB: 255, 0, 0)

	img = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)

	indices = np.where(image == 0)
	img[indices] = (255, 255, 255)

	indices = np.where(image == 1)
	img[indices] = (0, 0, 255)

	indices = np.where(image == 2)
	img[indices] = (0, 255, 255)

	indices = np.where(image == 3)
	img[indices] = (0, 255, 0)

	indices = np.where(image == 4)
	img[indices] = (255, 255, 0)

	indices = np.where(image == 5)
	img[indices] = (255, 0, 0)

	return img

def apply_DGlands_colormap(image):
	# DG lands colormap
	# Urban land: 0,255,255 
	# Agriculture land: 255,255,0
	# Rangeland: 255,0,255
	# Forest land: 0,255,0
	# Water: 0,0,255
	# Barren land: 255,255,255
	# Unknown: 0,0,0

	img = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)

	indices = np.where(image == 0)
	img[indices] = (0, 255, 255)

	indices = np.where(image == 1)
	img[indices] = (255, 255, 0)

	indices = np.where(image == 2)
	img[indices] = (255, 0, 255)

	indices = np.where(image == 3)
	img[indices] = (0, 255, 0)

	indices = np.where(image == 4)
	img[indices] = (0, 0, 255)

	indices = np.where(image == 5)
	img[indices] = (255, 255, 255)

	indices = np.where(image == 6)
	img[indices] = (0, 0, 0)

	return img




print "Converting RGB GT to 8-bit GT image"
source_path = sys.argv[1]
destination_path = sys.argv[2]
dataset = sys.argv[3]

if not os.path.isfile(source_path):
	print "ERROR: file not found --->", source_path
else:

	img = cv2.imread(source_path)[:,:,::-1]  ### BGR->RGB

	if img.shape[2] != 3:
		print 'skipping ', source_path
	else:
		target = 7*np.ones((img.shape[0], img.shape[1]), dtype = np.uint8)

	if dataset == 'potsdam':
		indices = np.all(img == (255, 255, 255), axis=-1)
		target[indices] = 0

		indices = np.all(img == (0, 0, 255), axis=-1)
		target[indices] = 1

		indices = np.all(img == (0, 255, 255), axis=-1)
		target[indices] = 2

		indices = np.all(img == (0, 255, 0), axis=-1)
		target[indices] = 3

		indices = np.all(img == (255, 255, 0), axis=-1)
		target[indices] = 4

		indices = np.all(img == (255, 0, 0), axis=-1)
		target[indices] = 5
	elif dataset == 'deepglobe_lands':
		indices = np.all(img == (0, 255, 255), axis=-1)
		target[indices] = 0

		indices = np.all(img == (255, 255, 0), axis=-1)
		target[indices] = 1

		indices = np.all(img == (255, 0, 255), axis=-1)
		target[indices] = 2

		indices = np.all(img == (0, 255, 0), axis=-1)
		target[indices] = 3

		indices = np.all(img == (0, 0, 255), axis=-1)
		target[indices] = 4

		indices = np.all(img == (255, 255, 255), axis=-1)
		target[indices] = 5

		indices = np.all(img == (0, 0, 0), axis=-1)
		target[indices] = 6


		
	print "writing", destination_path
	cv2.imwrite(destination_path, target)