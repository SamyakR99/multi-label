#### Continuing form  pascal-voc2012/dataloader.py

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


############################ LOAD CLIP MODEL ############################

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(clip.available_models())
clip_model, preprocess = clip.load('ViT-B/32', device)
clip_model = clip_model.float()


######################### SAVING CLIP FEATURES ############################
################## RUN ONLY ONCE AND LOAD THE SAVED ONES ##################
def get_features(dataloader):
    all_features_batches = []
    all_labels_batches = []
    for images, labels in dataloader:
        features = clip_model.encode_image(images.to(device))
        all_features_batches.append(features.detach())
        all_labels_batches.append(labels)
    return all_features_batches, all_labels_batches

train_features, train_labels = get_features(train_dataloader)

val_features, val_labels = get_features(val_dataloader)

train_features_path = "/home/samyakr2/multilabel/ARK/pascal_train_clip_features_vit14.pt"
train_labels_path = '/home/samyakr2/multilabel/ARK/pascal_train_clip_labels_vit14.pt'
val_features_path = "/home/samyakr2/multilabel/ARK/pascal_val_clip_features_vit14.pt"
val_labels_path = '/home/samyakr2/multilabel/ARK/pascal_val_clip_labels_vit14.pt'
# Save the tensor to file
torch.save(train_features, train_features_path)
torch.save(train_labels, train_labels_path)

torch.save(val_features, val_features_path)
torch.save(val_labels, val_labels_path)


############################ LOADING CLIP FEATURES ############################
train_features = torch.load(train_features_path)
train_labels = torch.load(train_labels_path)
val_features = torch.load(val_features_path)
val_labels = torch.load(val_labels_path)
############################ ADAPTER CLASS ############################

class adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(adapter, self).__init__()
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
        out = self.sigmoid(out)
#         out = self.softmax(out)
        return out


loaded_model = load_model(model_path)
loaded_model = load_model(model_path)


############################ PREP FOR TRAIN  ############################
input_size = train_features[0].size(1)  
hidden_size = 100  # Define the size of the hidden layer
num_classes = len(train_labels[0][0])  # Assuming labels_batches is a list of lists of labels

print(input_size,hidden_size,num_classes)

# Initialize the model
model = adapter(input_size, hidden_size, num_classes).to(device)

# # Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multilabel classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

############################ TRAIN LOOP  ############################
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for features_batch, labels_batch in zip(train_features, train_labels):
        # Flatten features batch
        features_batch = features_batch.view(features_batch.size(0), -1)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels_batch, dtype=torch.float32)
        # Forward pass
        outputs = model(features_batch.to(device))

        # Compute loss
        loss = criterion(outputs, labels_tensor.to(device))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Print loss for the epoch
#     if epoch %100 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")


############################ TEST LOOP  ############################
# Define a function for testing the model
def test_model(model, criterion, features_batches, labels_batches, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for features_batch, labels_batch in zip(features_batches, labels_batches):
            # Move batch to device
            features_batch = features_batch.to(device)
            labels_tensor = torch.tensor(labels_batch, dtype=torch.float32).to(device)

            # Flatten features batch
            features_batch = features_batch.view(features_batch.size(0), -1)

            # Forward pass
            outputs = model(features_batch)

            # Compute loss
            loss = criterion(outputs, labels_tensor)

            test_loss += loss.item()

    # Average test loss
    avg_test_loss = test_loss / len(features_batches)
    print(f"Test Loss: {avg_test_loss}")

# Call the test function after training


test_model(model, criterion, val_features, val_labels, device)
