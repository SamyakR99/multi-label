import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO


catIds = coco.getCatIds()
imgIds = coco.getImgIds(catIds=catIds)

img_lab_dict = {}
indices_to_remove = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]

for idx, file in enumerate(os.listdir('/home/samyakr2/multilabel/data/coco/val2017/')):
    lab_vector = np.zeros(90)
    imgIds = coco.getImgIds(imgIds = [int(file.strip('0')[:-4])])
    annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        lab_vector[ann['category_id']-1]= 1
    lab_vector = np.delete(lab_vector, indices_to_remove)
    img_lab_dict[file] = lab_vector


file_path = 'val_labels.npy'
np.save(file_path, img_lab_dict) 
