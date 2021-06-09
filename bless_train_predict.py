#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, random

import torch
from torch import nn
from torch import optim
from torch import tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
import PIL
from PIL import Image
import argparse

#model architecture
arch = {'vgg16':25088,
        'densenet121':102,
        'alexnet':4096}

def data_load(data_loc = './flowers'):
    data_dir = data_loc
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets  
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
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True) #maybe add shuffle to see if you would get better results
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle=True)
    
    return trainloader, testloader, validloader, train_data, test_data, valid_data


def model_setup(arch1='vgg16',dropout=0.5,hidden_layer=4096, lr=0.0003):
    if arch1 == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch1 == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch1 == "densenet121":
        model = models.densenet121(pretrained=True)    
    else:
        print('Please insert alexnet or densenet pre-trained models or it defaults to vgg16')
        model = models.vgg16(pretrained=True)
    
    #load a pre-trained network 
    #model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier =  nn.Sequential(OrderedDict([
                               ('input', nn.Linear(arch.get(arch1), hidden_layer)),
                               ('relu', nn.ReLU()),
                               ('dropout', nn.Dropout(dropout)),
                               ('hidden_layer', nn.Linear(hidden_layer, 102)),
                               ('output', nn.LogSoftmax(dim=1))
                                ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()

    #training the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr) 

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")          
    model.to(device);
    
    return model, criterion, optimizer

def training_network(model, criterion, optimizer,trainloader, validloader,epochs=1):
    steps = 0
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        running_loss = 0    
               
        for inputs, labels in trainloader: #adjusted trainloader to loader due to function call
                        
            steps += 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
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
            

def checkpoint_save(model,train_data, arch1='vgg16',epochs=1, dropout=0.5,hidden_layer=4096, lr=0.0003 ):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch1': arch1,
        'epoch': epochs,
        'dropout': dropout,
        'learning_rate': lr,
        'hidden_layer': hidden_layer,
        'model_state_dict': model.state_dict(),
        'class_to_idx':  model.class_to_idx}
                   
    #saving checkpoint in checkpoint.pth
    torch.save(checkpoint, 'checkpoint.pth')
 
def checkpoint_load(path='checkpoint.pth'): 
              checkpoint = torch.load(path)
              arch1 = checkpoint['arch1']
              hidden_layer = checkpoint['hidden_layer']
              dropout = checkpoint['dropout']
              lr = checkpoint['learning_rate']
              
              model, criterion, optimizer = model_setup(arch1,dropout,hidden_layer, lr)
              model.load_state_dict(checkpoint['model_state_dict'])
              model.class_to_idx= checkpoint['class_to_idx']
              return model  
              

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #open the image 
    image = PIL.Image.open(image)
    
    #resize & crop the image
    image = image.resize((256,256)).crop((0,0,224,224))
    
    #convert color channels of the images to float
    np_image = np.array(image)
    np_image = np_image/255 #color channels are typically encoded as integers 0-255 but model expects floats 0-1
    
    #normalize the images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std # substract mean from each color channel and divide by std
    
    #transpose the color channel to 1st dim - 
    #when loading image has the following dimension (width,height, color):width is 0, height is 1 and color is 2
    np_image = np_image.transpose((2,0,1))
    
    
    #return torch.from_numpy(np_image)
    return torch.FloatTensor(np_image)# FLOAT TENSOR MUST BE RETURNED OR ALTERNATIVELY CONVERTED TO LATER IN THE PROJECT

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use GPU if it's available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              
    #Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze(0) # FIRST DIMENSION - BATCH SIZE - MUST BE UNSQUEEZED SINCE MODEL EXPECTS 4 DIMENSIONAL TENSOR
    image = image.to('cpu') # MODEL AND IMAGE BOTH MUST BE SET TO THE SAME DEVICE
    model = model.to('cpu')
    
    print(image.device) # DEVICE TEST
    print(next(model.parameters()).is_cuda) # DEVICE TEST
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output) 
        
        #Get the top  𝐾  largest values in a tensor use
        top_p, top_class = ps.topk(topk)
        
        #map idx_to_class to the model
        #model.class_to_idx = train_data.class_to_idx
        #model.class_to_idx = checkpoint_save(epochs)[0]
        idx_to_class = {val: key for key, val in model.class_to_idx.items()} #new removed
        
        #convert from these indices to the actual class labels
        top_p = top_p.squeeze().tolist()
        #classes = [model.idx_to_class[idx] for idx in top_class[0].tolist()] 
        classes = [idx_to_class[idx] for idx in top_class[0].tolist()] #new removed
        #labels = [cat_to_name[i] for i in top_class]
        
    return top_p,classes # MUST RETURN THE CLASSES, VARIABLE NAMES BROUGHT IN LINE HERE - removed classes