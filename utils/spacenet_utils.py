import cv2
import sys
import csv
import os.path
import numpy as np
import tifffile as tiff
from scipy import ndimage

def get_line_string_dict(path):
	file=open( path, "r")
	linestrings = {}
	reader = csv.reader(file)
	for line in reader:
		line[0] = 'RGB-PanSharpen_' + line[0]
		l = []
		if line[1] == ' LINESTRING EMPTY':
			l = None
		else:
			for f in range(1, len(line)):
				k = line[f].split(' ')
				l+= [[int(float(k[1])), int(float(k[2]))]]

		if line[0] in linestrings:
			linestrings[line[0]] += [l]
		else:
			linestrings[line[0]] = [l]

	return linestrings


list_file = sys.argv[1]
source_img_root_path = sys.argv[2]
target_img_root_path = sys.argv[3]
target_gt_root_path = sys.argv[4]
linestrings_path = sys.argv[5]
stride = int(sys.argv[6])

linestrings = get_line_string_dict(linestrings_path)
clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

image_list = [line.rstrip('\n') for line in open(list_file)]
print len(image_list)

for img_name in image_list:
	print 'processing', img_name
	img= tiff.imread(source_img_root_path + img_name + '.tif')
	red = np.asarray(img[:,:,0],dtype=np.float)
	green = np.asarray(img[:,:,1],dtype=np.float)
	blue = np.asarray(img[:,:,2],dtype=np.float)
	
	red_ = 255.0 * ((red-np.min(red))/(np.max(red) - np.min(red) + 1e-12))
	green_ = 255.0 * ((green-np.min(green))/(np.max(green) - np.min(green) + 1e-12))
	blue_ = 255.0 * ((blue-np.min(blue))/(np.max(blue) - np.min(blue) + 1e-12))
	
	img_rgb = np.zeros((1300,1300,3),dtype=np.uint8)
	img_rgb[:,:,0] = clahe.apply(np.asarray(red_,dtype=np.uint8))
	img_rgb[:,:,1] = clahe.apply(np.asarray(green_,dtype=np.uint8))
	img_rgb[:,:,2] = clahe.apply(np.asarray(blue_,dtype=np.uint8))
	
	seg_gt = np.ones((1300,1300),dtype=np.float)
	if img_name in linestrings:
		lines = linestrings[img_name]
		if lines[0] is not None:
			for l in lines:
				for points in range(1, len(l)):
					cv2.line(seg_gt, (l[points][0], l[points][1]), (l[points-1][0], l[points-1][1]), 0, 1)
	else:
		print 'ERROR: couldn\'t find linestrings for -> ', img_name

	std = 15.0
	dist = ndimage.distance_transform_edt(seg_gt)
	seg_gt = np.exp(-0.5*(dist/std)**2)

	seg_gt = 255 * seg_gt

	print 'writing', target_img_root_path + img_name + '.jpg'
	cv2.imwrite(target_img_root_path + img_name + '.jpg', img_rgb[:,:,::-1])

	print 'writing', target_gt_root_path + img_name + '.png'
	cv2.imwrite(target_gt_root_path + img_name + '.png', seg_gt)

	for rows in range(0, img_rgb.shape[0], stride):
		for cols in range(0, img_rgb.shape[1], stride):
			if rows+650 > img_rgb.shape[0] or cols+650 > img_rgb.shape[1]:
				continue
			cv2.imwrite(target_img_root_path + img_name + '_' + str(rows) + '_' + str(cols) + '.jpg', img_rgb[rows:rows+650, cols:cols+650, ::-1])
			cv2.imwrite(target_gt_root_path + img_name + '_' + str(rows) + '_' + str(cols) + '.png', seg_gt[rows:rows+650, cols:cols+650])
