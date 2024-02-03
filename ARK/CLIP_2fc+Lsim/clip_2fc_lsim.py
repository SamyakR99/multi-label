#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

torch.autograd.set_detect_anomaly(True)


# In[2]:


# device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(clip.available_models())
# clip_model, preprocess = clip.load('ViT-L/14', device)
# clip_model = clip_model.float()


# In[3]:


# labels_pascal = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# items_pascal = ["A photo of a " + item for item in labels_pascal]

# text = clip.tokenize(items_pascal).to(device)
# text_features = clip_model.encode_text(text)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# text_features_path = '/home/samyakr2/multilabel/ARK/new_idea/pascal_labels_features.pt'
# torch.save(text_features, text_features_path)
# ### For code to obtain .npy file below look at ll_sim.py in github
# # y = np.load('/home/samyakr2/multilabel/data/pascal/pascal_ll_sim_vit14.npy')



# In[4]:


class projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(projector, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim),
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
#         out = self.relu(out)
        return out

class projector2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(projector2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim, bias=False),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = 0.15*x + 0.85*out
        out = self.relu(out)
        out = self.fc2(out)
        return out


# def Projector(args, embedding):
#     mlp_spec = f"{embedding}-{args.mlp}"
#     layers = []
#     f = list(map(int, mlp_spec.split("-")))
#     for i in range(len(f) - 2):
#         layers.append(nn.Linear(f[i], f[i + 1]))
#         layers.append(nn.BatchNorm1d(f[i + 1]))
#         layers.append(nn.ReLU(True))
#     layers.append(nn.Linear(f[-2], f[-1], bias=False))
#     return nn.Sequential(*layers)


# In[5]:


device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
text_features_path = '/home/samyakr2/multilabel/ARK/new_idea/pascal_labels_features.pt'
text_features = torch.load(text_features_path)

# in_s = text_features.shape[1]  
# hs =text_features.shape[1]    # Define the size of the hidden layer
# nc = text_features.shape[1]  

# model_text = projector(in_s, hs, nc).to(device)
# model_text2 = projector2(input_size, hidden_size, num_classes).to(device)
# outputs_text = model_text(text_features.to(device))
# # outputs_text2 = model_text2(text_features.to(device))

# # with torch.no_grad():
# similarity_text = (outputs_text @ outputs_text.T) 
# # similarity_text2 = (outputs_text2 @ outputs_text2.T) 


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(clip.available_models())
clip_model, preprocess = clip.load('RN50x64', device)
clip_model = clip_model.float()

train_features_path = "/home/samyakr2/multilabel/ARK/pascal_train_clip_features_vit14.pt"
train_labels_path = '/home/samyakr2/multilabel/ARK/pascal_train_clip_labels_vit14.pt'
val_features_path = "/home/samyakr2/multilabel/ARK/pascal_val_clip_features_vit14.pt"
val_labels_path = '/home/samyakr2/multilabel/ARK/pascal_val_clip_labels_vit14.pt'

train_features = torch.load(train_features_path)
train_labels = torch.load(train_labels_path)
val_features = torch.load(val_features_path)
val_labels = torch.load(val_labels_path)


# In[30]:


import os

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class clip_2fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(clip_2fc, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False)
        )
        
#         self.fc1 = nn.Linear(input_dim, hidden_dim),
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
#         out = self.sigmoid(out)
        return out
    
input_size = train_features[0].size(1)  
hidden_size = 100  # Define the size of the hidden layer
num_classes = len(train_labels[0][0])  # Assuming labels_batches is a list of lists of labels

in_s = text_features.shape[1]  
hs = 200  # Define the size of the hidden layer
nc = 100  

print(in_s)
# Initialize the model
model = clip_2fc(input_size, hidden_size, num_classes).to(device)
model_text = projector(in_s, hs, nc).to(device)

# # Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multilabel classification

params_to_optimize = list(model.parameters()) + list(model_text.parameters())
optimizer = torch.optim.Adam(params_to_optimize, lr=0.001)  # Adam optimizer with learning rate 0.001

# # Training loop

best_loss = float('inf')
num_epochs = 42

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for features_batch, labels_batch in zip(train_features, train_labels):
        # Flatten features batch
        features_batch = features_batch.view(features_batch.size(0), -1)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels_batch, dtype=torch.float32)#.clone().detach()
        # Forward pass
        outputs = model(features_batch.to(device))
        outputs_reshaped = outputs.unsqueeze(-1)
#         outputs_reshaped_normalized = F.normalize(outputs_reshaped, p=2, dim=1)
        
        
        outputs_text = model_text(text_features.to(device))
        similarity_text = (outputs_text @ outputs_text.T)
#         print(similarity_text.unsqueeze(0).expand(outputs.shape[0], -1, -1).shape)
#         print(outputs_reshaped.shape)
        normalized_similarity_text = F.normalize(similarity_text, p=2, dim=1)  # Normalize along the second dimension (rows)
        normalized_similarity_text = torch.clamp(normalized_similarity_text, min=0, max=1)  # Clamp values to be between 0 and 1

        result = outputs_reshaped * normalized_similarity_text.unsqueeze(0).expand(outputs.shape[0], -1, -1)

#         result = outputs_reshaped * similarity_text.unsqueeze(0).expand(outputs.shape[0], -1, -1)
        
        pred = result.sum(dim=1) / 16
#         print("pred shape", pred.shape)
        pred = torch.sigmoid(pred)
#         Compute loss
        loss = criterion(pred, labels_tensor.to(device))
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
#         print("=="*50)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_model_state_dict = model.state_dict()
        best_text_model_dict = model_text.state_dict()
    
    # Save the model every 20 epochs
    if (epoch + 1) % 20 == 0:
        torch.save(best_model_state_dict, f"/home/samyakr2/multilabel/ARK/new_idea/best_epoch_{epoch+1}.pth")
        torch.save(best_text_model_dict, f"/home/samyakr2/multilabel/ARK/new_idea/best_text_epoch_{epoch+1}.pth")


# In[31]:


import numpy as np
from sklearn.metrics import average_precision_score

# Define a function for testing the model
def test_model(model, model_text,text_features,criterion, features_batches, labels_batches, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():  # Disable gradient computation
        for features_batch, labels_batch in zip(features_batches, labels_batches):
            # Move batch to device
            features_batch = features_batch.to(device)
            labels_tensor = torch.tensor(labels_batch, dtype=torch.float32).to(device)

            # Flatten features batch
            features_batch = features_batch.view(features_batch.size(0), -1)

            # Forward pass
            outputs = model(features_batch)
            outputs_text = model_text(text_features.to(device))
#             similarity_text = (outputs_text @ outputs_text.T)
#             print(outputs)
            outputs_reshaped = outputs.unsqueeze(-1)
            
            
            result = outputs_reshaped * normalized_similarity_text.unsqueeze(0).expand(outputs.shape[0], -1, -1)
        
            pred = result.sum(dim=1) / 16
    #         print("pred shape", pred.shape)
#             pred = torch.sigmoid(pred)
#             loss = criterion(pred, labels_tensor)

            loss = criterion(torch.sigmoid(pred), labels_tensor)

            test_loss += loss.item()

            # Convert outputs and labels to numpy arrays
            outputs_np = torch.sigmoid(pred).cpu().detach().numpy()
            labels_np = labels_tensor.cpu().detach().numpy()

            all_outputs.append(outputs_np)
            all_labels.append(labels_np)

    # Concatenate outputs and labels
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Compute average precision score
    avg_precision = average_precision_score(all_labels, all_outputs, average='micro')

    # Average test loss
    avg_test_loss = test_loss / len(features_batches)
    print(f"Test Loss: {avg_test_loss}")
    print(f"Average Precision Score: {avg_precision}")

for i in range (20,200,20):
    best_model_state_dict = torch.load("/home/samyakr2/multilabel/ARK/new_idea/best_epoch_{}.pth".format(i))
    model.load_state_dict(best_model_state_dict)

    best_model_state_dict_text = torch.load("/home/samyakr2/multilabel/ARK/new_idea/best_text_epoch_{}.pth".format(i))
    model_text.load_state_dict(best_model_state_dict_text)

    test_model(model, model_text, text_features, criterion, val_features, val_labels, device)


# In[ ]:




