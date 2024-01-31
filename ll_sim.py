import os
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from itertools import permutations
from scipy.special import kl_div
import itertools
import numpy as np
import copy
import shutil


labels_pascal = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
items_pascal = ["A photo of a " + item for item in labels_pascal]

print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x4", device=device)

text = clip.tokenize(items_pascal).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
print(text_features.shape)
inter_text_sim = torch.tensor([]).to(device)
with torch.no_grad():
    similarity = (text_features @ text_features.T) #.softmax(dim=-1)
    inter_text_sim = torch.cat((inter_text_sim,similarity),dim = 0)
    
similarity_np = similarity.detach().cpu().numpy()
np.save('/home/samyakr2/multilabel/data/pascal/pascal_ll_sim_resnet.npy', similarity_np)

## Some examples printed
for i in range (len(labels_pascal)):
    val, idx = similarity[i].topk(5)
    idx_list = idx.tolist()
    selected_labels = [items_pascal[i] for i in idx_list]
    print(selected_labels)
    print(val)


################################################### FOR COCO ###############################################

coco_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","trafficlight","firehydrant","streetsign","stopsign","parkingmeter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eyeglasses","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball","kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket","bottle","plate","wineglass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tvmonitor","laptop","mouse","remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator","blender","book","clock","vase","scissors","teddybear","hairdrier","toothbrush","hairbrush"]
coco_items = ["A photo of a " + item for item in coco_labels]

text = clip.tokenize(items).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)
inter_text_sim = torch.tensor([]).to(device)
with torch.no_grad():
    similarity = (text_features @ text_features.T) #.softmax(dim=-1)
    inter_text_sim = torch.cat((inter_text_sim,similarity),dim = 0)

    
similarity_np = similarity.detach().cpu().numpy()
np.save('/home/samyakr2/multilabel/data/coco/coco_ll_sim_resnet.npy', similarity_np)


## Some examples printed
for i in range (len(coco_labels)):
    val, idx = similarity[i].topk(5)
    idx_list = idx.tolist()
    selected_labels = [coco_items[i] for i in idx_list]
    print(selected_labels)
    print(val)
