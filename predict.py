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

import bless_train_predict


parser = argparse.ArgumentParser(description='Prediction script')

parser.add_argument('--gpu', type=bool, help ='gpu available?',dest='gpu', default='gpu')
parser.add_argument('--image_input', default ='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg', action='store', type=str)
parser.add_argument('--checkpoint', default ='/home/workspace/ImageClassifier/checkpoint.pth', action='store', type=str )
parser.add_argument('--top_k', dest = 'top_k', default=5 ,action='store', type=int)
parser.add_argument('--category_names', action='store', default = 'cat_to_name.json', dest = 'category_names')

args = parser.parse_args()

image_path = args.image_input #path to image
topk = args.top_k # K most likely classes
path = args.checkpoint #path that loads the checkpoint and rebuilds the model

trainloader, testloader, validloader, train_data, test_data, valid_data= bless_train_predict.data_load()

#load the checkpoint
model = bless_train_predict.checkpoint_load(path)

#load in the mapping for integer encoded categories to actual flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#make predictions
top_p, classes = bless_train_predict.predict(image_path, model, topk) #probabilities and classes
print(top_p)
print(classes)







