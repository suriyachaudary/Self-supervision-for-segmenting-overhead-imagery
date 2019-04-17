# sample code from https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py

#!/usr/bin/env python

import collections

import numpy as np
import PIL.Image
import cv2
import torch
from torch.utils import data
import os

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def apply_color_map(image, c_map):
    canvas = 255*np.ones((image.shape[0], image.shape[1], 3), np.uint8)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            val = image[r,c]
            if val >= c_map.shape[0]:
                continue
            canvas[r,c,0] = c_map[val][0]
            canvas[r,c,1] = c_map[val][1]
            canvas[r,c,2] = c_map[val][2]
    return canvas


class context_inpainting_dataloader(data.Dataset):
    
    def __init__(self, img_root, image_list, split = 'train', suffix = '', 
                 mirror = True, resize = False, resize_shape = [256, 256], rotate = True,
                 crop = True, crop_shape = [128, 128], erase_shape = [16, 16], erase_count = 16):

        self.img_root = img_root
        self.split = split
        self.image_list = [line.rstrip('\n') for line in open(image_list)]
        self.img_suffix = None
        self.gt_suffix = None
        if suffix == 'potsdam' or suffix == 'spacenet':
            self.img_suffix = ''
        elif suffix == 'deepglobe_roads' or suffix == 'deepglobe_lands':
            self.img_suffix = ''

        self.mirror = mirror
        self.rotate = rotate
        self.resize = resize
        self.resize_shape = resize_shape
        
        self.crop = crop
        self.crop_shape = crop_shape
        self.erase_shape = erase_shape
        self.erase_count = erase_count


        self.mean_bgr = np.array([85.5517787014, 92.6691667083, 86.8147645556])
        self.std_bgr = np.array([32.8860206505, 31.7342205253, 31.5361127226])
        
        self.files = collections.defaultdict(list)
        for f in self.image_list:
            self.files[self.split].append({'img': f})
        
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        image_file_name = self.img_root + self.image_list[index] + self.img_suffix + '.jpg'
        
        image = None
        if os.path.isfile(image_file_name):
            image = cv2.imread(image_file_name)
        else:
            print('couldn\'t find image -> ', image_file_name)
            
        if self.mirror:                                                ### mirror image with probability of 0.5
            flip = torch.LongTensor(1).random_(0, 2)[0]*2-1
            image = image[:, ::flip, :]

        if self.rotate:                                               ### randomly rotate image
            choice = torch.LongTensor(1).random_(0, 4)[0]
            angles = [0, 90, 180, 270]
            angle = angles[choice]
            center = tuple(np.array(image.shape)[:2]/2)
            rot_mat = None
            rot_mat = cv2.getRotationMatrix2D(center,angle, 1 )
            image = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)

        if self.resize == True and torch.LongTensor(1).random_(0, 2)[0] == 1:        ### resize image with probability of 0.5
            if self.resize_shape[0] != image.shape[0] or self.resize_shape[1] != image.shape[1]:
                image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        
        if self.crop:
            image = self.get_random_crop(image, self.crop_shape)

        mask = np.ones((image.shape[0], image.shape[1], 3), np.uint8)
        input_ = None

        if self.erase_count == 1:                             ### erase a patch in the center of image
            offset = (image.shape[0] - erase_shape[0])/2
            end = offset+erase_shape[0]
            mask[offset:end, offset:end, :] = 0
            
        else:   
            for c_ in range(self.erase_count):
                row = torch.LongTensor(1).random_(0, image.shape[0]-self.erase_shape[0]-1)[0]
                col = torch.LongTensor(1).random_(0, image.shape[1]-self.erase_shape[1]-1)[0]
                
                mask[row:row+self.erase_shape[0], col:col+self.erase_shape[1], :] = 0
                                
        input_, mask, image = self.transform(mask, image)

        return input_, mask, image

    def transform(self, mask, image):
        image = image.astype(np.float64)
        image -= self.mean_bgr
               
        input_ = image.copy()                        
        
        image[:,:,0] /= 3*self.std_bgr[0]
        image[:,:,1] /= 3*self.std_bgr[1]
        image[:,:,2] /= 3*self.std_bgr[2]

        index_ = image > 1
        image[index_] = 1
        index_ = image < -1
        image[index_] = -1
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        
        input_ = input_.transpose(2, 0, 1)
        input_ = torch.from_numpy(input_.copy()).float()                        
        
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask.copy())                        

        return input_, mask, image

    def get_random_crop(self, im, crop_shape):
        """
        crop shape is of the format: [rows cols]
        """
        r_offset = torch.LongTensor(1).random_(0, im.shape[0] - crop_shape[0] + 1)[0]
        c_offset = torch.LongTensor(1).random_(0, im.shape[1] - crop_shape[1] + 1)[0]
        
        if im.shape[0] == crop_shape[0]:
            r_offset = 0
        if im.shape[1] == crop_shape[1]:
            c_offset = 0

        crop_im = im[r_offset: r_offset + crop_shape[0], c_offset: c_offset + crop_shape[1], :]

        return crop_im


class segmentation_data_loader(data.Dataset):
    
    def __init__(self, img_root, gt_root, image_list, split = 'train', mirror = True,
                     resize = False, resize_shape = [256, 256],  suffix='', out='',
                     rotate = True, crop = False, crop_shape=[256, 256], image_backend = 'cv2'):
        
        ### paths
        self.img_root = img_root
        self.gt_root = gt_root

        self.split = split
        self.backend = image_backend
        self.out = out
        
        ### list of all images
        self.image_list = [line.rstrip('\n') for line in open(image_list)]
        
        
        ### augmentations
        self.mirror = mirror
        self.rotate = rotate
        
        self.resize = resize
        self.resize_shape = resize_shape
        
        self.crop = crop
        self.crop_shape = crop_shape
        
        self.img_suffix = None
        self.gt_suffix = None
        
        if suffix == 'potsdam' or suffix == 'spacenet':
            self.img_suffix = ''
            self.gt_suffix = ''
        elif suffix == 'deepglobe_roads' or suffix == 'deepglobe_lands':
            self.img_suffix = ''
            self.gt_suffix = ''

        ### pre-processing inputs
        self.mean_bgr = np.array([85.5517787014, 92.6691667083, 86.8147645556])
        self.std_bgr = np.array([32.8860206505, 31.7342205253, 31.5361127226])
        
        self.files = collections.defaultdict(list)
        for f in self.image_list:
            self.files[self.split].append({'img': f,'lbl':f})
        
    def __len__(self):
        return len(self.files[self.split])


    def __getitem__(self, index):
        
        image_file_name = self.img_root + self.image_list[index] + self.img_suffix + '.jpg'
        seg_gt_name = self.gt_root + self.image_list[index] + self.gt_suffix + '.png'
        
        ### read image
        image = None
        if os.path.isfile(image_file_name):
            image = cv2.imread(image_file_name)
        else:
            print('couldn\'t find image -> ', image_file_name)
        
            
        ### read segmentation gt as greyscale image
        seg_gt = None
        if os.path.isfile(seg_gt_name):
            if self.backend == 'cv2':
                seg_gt = cv2.imread(seg_gt_name, 0)
            else:
                seg_gt = np.array(PIL.Image.open(seg_gt_name), dtype = np.uint8)
    
        else:
            print('couldn\'t find segmentation gt ->', seg_gt_name)
        if self.out == 'seg':
            seg_gt = (seg_gt/255.0 > 0.4).astype(np.uint8)
        elif self.out =='heatmap':
            seg_gt = seg_gt

        
        if self.crop == True:
            if self.crop_shape[0] > image.shape[0] or self.crop_shape[1] > image.shape[1]:
                ### apply zero pad when image is smaller than crop_shape
                pad_im, pad_seg_gt = self.pad_before_crop(image, seg_gt, self.crop_shape, self.ignore_label)
                image = pad_im
                seg_gt = pad_seg_gt
            
            ### get random crop
            image, seg_gt = self.get_random_crop(image, seg_gt, self.crop_shape)
        
        ### apply mirroring
        if self.mirror == 1:
            flip = torch.LongTensor(1).random_(0, 2)[0]*2-1
            image = image[:, ::flip, :]
            seg_gt = seg_gt[:, ::flip]

        if self.rotate:
            choice = torch.LongTensor(1).random_(0, 4)[0]
            angles = [0, 90, 180, 270]
            angle = angles[choice]
            center = tuple(np.array(image.shape)[:2]/2)
            rot_mat = None
            rot_mat = cv2.getRotationMatrix2D(center,angle, 1 )
            image = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
            seg_gt = cv2.warpAffine(seg_gt, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
            
        # need to resize input?
        if self.resize == True:
            if self.resize_shape[0] != image.shape[0] or self.resize_shape[1] != image.shape[1]:
                image = cv2.resize(image, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
                seg_gt = cv2.resize(seg_gt, (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)


        return self.transform(image, seg_gt)

    
    def pad_before_crop(self, im, seg_gt, crop_shape, ignore_label):
        # print "padding image"
        rows = im.shape[0]
        cols = im.shape[1]
        r_offset = 0
        c_offset = 0
        if im.shape[0] < crop_shape[0]:
            rows = crop_shape[0]
            r_offset = int((crop_shape[0] - im.shape[0])/2)   ### place image in center of canvas
        if im.shape[1] < crop_shape[1]:
            cols = crop_shape[1]
            c_offset = int((crop_shape[1] - im.shape[1])/2)   ### place image in center of canvas

        img_canvas = np.zeros((rows, cols, 3), np.uint8)
        img_canvas[r_offset : r_offset+im.shape[0], c_offset : c_offset + im.shape[1], : ] = im

        seg_canvas = ignore_label*np.ones((rows, cols), np.uint8)
        seg_canvas[r_offset : r_offset+im.shape[0], c_offset : c_offset + im.shape[1]] = seg_gt

        return img_canvas, seg_canvas
 
    
    def get_random_crop(self, im, seg_gt, crop_shape):
        """
        crop shape is of the format: [rows cols]
        im is padded if dimension of im is less than specified crop shape
        """
        r_offset = torch.LongTensor(1).random_(0, im.shape[0] - crop_shape[0] + 1)[0]
        c_offset = torch.LongTensor(1).random_(0, im.shape[1] - crop_shape[1] + 1)[0]
        
        if im.shape[0] == crop_shape[0]:
            r_offset = 0
        if im.shape[1] == crop_shape[1]:
            c_offset = 0

        crop_im = im[r_offset: r_offset + crop_shape[0], c_offset: c_offset + crop_shape[1], :]
        crop_seg = seg_gt[r_offset: r_offset + crop_shape[0], c_offset: c_offset + crop_shape[1]]

        return crop_im, crop_seg

    def transform(self, img, lbl):
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        return img, lbl
