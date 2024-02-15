#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Define transforms for data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the dataset
data_dir = '../computer_vision_Hai_Hiep/patchcore-inspection/mvtec_anomaly_detection/bottle'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

# Load pre-trained WideResNet model
pretrained_model = models.wide_resnet50_2(pretrained=True)
num_features = pretrained_model.fc.in_features

# Replace the last fully connected layer with a binary classification layer
pretrained_model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

# Train the model
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in (range(num_epochs)):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 50)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)  # Convert labels to float and unsqueeze to match model output shape

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

    return model

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_model = pretrained_model.to(device)
trained_model = train_model(pretrained_model, criterion, optimizer, num_epochs=10)

