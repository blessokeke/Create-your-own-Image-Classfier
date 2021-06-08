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

import train_main

parser = argparse.ArgumentParser(description='Prediction script')

parser.add_argument('--gpu', type=bool, help ='gpu available?')
parser.add_argument('--data_dir', help ='data directory', default = 'flowers')
parser.add_argument('--image_input', default ='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg', action='store', type=str)
parser.add_argument('--checkpoint', default ='/home/workspace/ImageClassifier/checkpoint.pth', action='store', type=str )
parser.add_argument('--top_k', dest = 'top_k', default=5 ,action='store')
parser.add_argument('--category_names', action='store', default = 'cat_to_name.json')
parser.add_argument('--arch', type=str, help ='vgg16 model architecture', default = models.vgg16(pretrained=True))

args = parser.parse_args()


image_path = args.image_input #path to image
top_p = args.top_k # K most likely classes
path = args.checkpoint #path that loads the checkpoint and rebuilds the model
model = args.arch

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#load in the mapping for integer encoded categories to actual flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#process image function  
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
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
    
    # TODO: Implement the code to predict the class from an image file
    
     
    image = process_image(image_path)
    image = image.unsqueeze(0) # FIRST DIMENSION - BATCH SIZE - MUST BE UNSQUEEZED SINCE MODEL EXPECTS 4 DIMENSIONAL TENSOR
    image = image.to('cpu') # MODEL AND IMAGE BOTH MUST BE SET TO THE SAME DEVICE, SEE PRINTOUTS BELOW FOR DEMO
    model = model.to('cpu')
    
    print(image.device) # DEVICE TEST
    print(next(model.parameters()).is_cuda) # DEVICE TEST
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output) 
        
        #Get the top  𝐾  largest values in a tensor use
        top_p, top_class = ps.topk(topk)
        
        #map idx_to_class to the model
        idx_to_class = {val: key for key, val in model.class_to_idx.items()} #new
        
        #convert from these indices to the actual class labels
        top_p = top_p.squeeze().tolist()
        #classes = [model.idx_to_class[idx] for idx in top_class[0].tolist()] 
        classes = [idx_to_class[idx] for idx in top_class[0].tolist()] #new
        
    return top_p,classes # MUST RETURN THE CLASSES, VARIABLE NAMES BROUGHT IN LINE HERE

#predict flower name from an image along with the probability of that name
top_p, classes = predict(test_dir+'/10/image_07090.jpg', model)
print(top_p)
print(classes)