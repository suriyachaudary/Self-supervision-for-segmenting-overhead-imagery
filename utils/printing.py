#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt



def training_curves_loss(train_loss_hist, val_loss_hist):
   
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss_hist,  'C0', alpha=0.5, label='Train loss')
    ax1.plot(val_loss_hist, 'C1', alpha=0.5, label='Val loss', linewidth=2.0)
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('loss')
    fig.tight_layout()
    plt.show()


def segmentation_training_curves_loss(train_loss_hist, val_loss_hist, train_iou, val_iou):
    fig, ax1 = plt.subplots()
    ax1.plot(train_loss_hist,  'C0', alpha=0.5, label='Train loss')
    ax1.plot(val_loss_hist, 'C1', alpha=0.5, label='Val loss', linewidth=2.0)
    
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel('loss')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(train_iou,  'C0--', alpha=0.5, label='train_iou')
    ax2.plot(val_iou, 'C1--', alpha=0.5, label='val_iou', linewidth=2.0)
    ax2.set_ylabel('mIoU')
    ax2.tick_params('y')
    ax1.legend(loc='upper center')
    ax2.legend(loc='best')
    fig.tight_layout()
    plt.show()
    
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
    