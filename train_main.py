import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
import PIL
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--gpu', type=bool, help ='gpu available?')
parser.add_argument('--data_dir', help ='data directory', default = 'flowers')
parser.add_argument('--arch', type=str, help ='vgg16 model architecture', default = models.vgg16(pretrained=True))
parser.add_argument('--save_dir', dest = 'save_dir', type=str, help ='directory for saved model', default = './checkpoint.pth')
parser.add_argument('--learning_rate', type=float, help ='learning rate', default = 0.0003)
parser.add_argument('--hidden_units', type=int, action = 'store', help ='hidden units #', default = 4096, dest = 'hidden_units')
parser.add_argument('--epochs', type=int, action = 'store', help ='epochs #', default = 1)
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
path = args.save_dir
epochs = args.epochs
lr = args.learning_rate

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
#data_transforms = 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
#image_datasets = 
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True) #maybe add shuffle to see if you would get better results
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle=True)

#label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
#load a pre-trained network 
model = models.vgg16(pretrained=True)    

#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout

# turn off gradients of my model so I don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier =  nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 #nn.Linear(4096, 256),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.5),
                                 nn.Linear(4096, 102),
                                 nn.LogSoftmax(dim=1))
    
model.classifier = classifier

#Train the classifier layers using backpropagation using the pre-trained network to get the features
criterion = nn.NLLLoss()

#training the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003) #lr=learning rate - reduce to see if accuracy will improve

model.to(device);

#Track the loss and accuracy on the validation set to determine the best hyperparameters
epochs = 1 #5
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
#define checkpoint and save it
checkpoint = {
            'epoch': epochs,
            'output_size': 102,
            'classifier': model.classifier,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx':  model.class_to_idx}
        
#saving checkpoint in checkpoint.pth
torch.save(checkpoint, 'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def checkpoint_load(path):
    model = models.vgg16(pretrained=True)
    checkpoint = torch.load(path)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx= checkpoint['class_to_idx']
    
    return model

