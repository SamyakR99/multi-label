import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class VOCDataset(Dataset):
    def __init__(self, image_names, label_names, image_dir, transforms=None):
        self.image_names = image_names
        self.label_names = label_names
        self.transforms = transforms
        self.image_dir = image_dir
#         self.label_dir = label_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        label_name = self.label_names[idx]

        image = Image.open(img_name).convert("RGB")
#         label = np.load(label_name)

        if self.transforms:
            image = self.transforms(image)

        return image, label_name

# Load filenames
train_image_names = np.load('/home/samyakr2/multilabel/data/pascal/formatted_train_images.npy')
train_label_names = np.load('/home/samyakr2/multilabel/data/pascal/formatted_train_labels.npy')
val_image_names = np.load('/home/samyakr2/multilabel/data/pascal/formatted_val_images.npy')
val_label_names = np.load('/home/samyakr2/multilabel/data/pascal/formatted_val_labels.npy')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


path_to_images = '/home/samyakr2/multilabel/data/pascal/VOCdevkit/VOC2012/JPEGImages'
train_dataset = VOCDataset(image_names=train_image_names, label_names=train_label_names, 
                     image_dir=path_to_images, transforms=transform)

val_dataset = VOCDataset(image_names=val_image_names, label_names=val_label_names, 
                     image_dir=path_to_images, transforms=transform)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
