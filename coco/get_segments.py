%matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

import os
from PIL import Image
import requests

phase = 'train'
read_dictionary = np.load(phase+'_labels.npy',allow_pickle='TRUE').item()
y = read_dictionary.keys()


dire = '/home/samyakr2/multilabel/data/coco'
annFile='{}/annotations/instances_{}.json'.format(dire,phase+'2017') ## for val
coco=COCO(annFile)

####################### NOTE: THIS IS ONYL FOR ONE IMAGE ##############################################
####################### As shown in the code below next(iter) take only the first item ########################
first_element = next(iter(read_dictionary.keys()))
int(first_element.split('.')[0])


import cv2
import numpy as np
import matplotlib.pyplot as plt

catIds = coco.getCatIds()
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [int(first_element.split('.')[0])])
img = cv2.imread('/home/samyakr2/multilabel/data/coco/'+phase+'2017/'+first_element)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
anns_ids = coco.getAnnIds(imgIds=[int(first_element.split('.')[0])], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(anns_ids)
anns_img = np.zeros((img.shape[0], img.shape[1]))

for ann in anns:
    anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
    
image = anns_img
unique_pixel_values = np.unique(image)
unique_pixel_values = unique_pixel_values[unique_pixel_values > 0]
num_unique_pixels = len(unique_pixel_values)

segmented_masks = []

fig, axes = plt.subplots(1, num_unique_pixels, figsize=(15, 3))

for i, pixel_value in enumerate(unique_pixel_values):
    # Create an image with only the current unique pixel value
    unique_image = np.zeros_like(image)
    unique_image[image == pixel_value] = pixel_value
    segmented_masks.append(unique_image)
    # Plot the unique image
    axes[i].imshow(unique_image, cmap='gray')
    axes[i].set_title(f"Pixel Value: {pixel_value}")
    axes[i].axis('off')
    

plt.tight_layout()
plt.show()


for i in range (len(segmented_masks)):
    segmented_masks[i] = (segmented_masks[i] > 0).astype(np.uint8)
    masked_image = img * segmented_masks[i][:, :, np.newaxis]



    # Plot the masked image
    plt.imshow(img)
    plt.show()
    plt.imshow(masked_image)
    plt.axis('off')  # Hide axis
    plt.show()
