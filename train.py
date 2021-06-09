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

parser = argparse.ArgumentParser(description='Training script')

parser.add_argument('--gpu', type=bool, help ='gpu available?',dest='gpu', default='gpu')
parser.add_argument('--data_dir', help ='data directory', default = './flowers/')
parser.add_argument('--arch', type=str, help ='vgg16 model architecture', default = 'vgg16')
parser.add_argument('--save_dir', dest = 'save_dir', type=str, help ='directory for saved model', default = './checkpoint.pth')
parser.add_argument('--learning_rate', type=float, help ='learning rate', default = 0.0003)
parser.add_argument('--hidden_units', type=int, action = 'store', help ='hidden units #', default = 4096, dest = 'hidden_units')
parser.add_argument('--dropout', action = 'store', help ='drop out', default = 0.5, dest = 'dropout')
parser.add_argument('--epochs', type=int, action = 'store', help ='epochs #', default = 1)
args = parser.parse_args()

data_loc = args.data_dir
path = args.save_dir
epochs = args.epochs
lr = args.learning_rate
arch1 = args.arch
dropout = args.dropout
hidden_layer = args.hidden_units

trainloader, testloader, validloader, train_data, test_data, valid_data= bless_train_predict.data_load(data_loc)


model, criterion, optimizer = bless_train_predict.model_setup(arch1, dropout, hidden_layer, lr)


bless_train_predict.training_network(model, criterion, optimizer,trainloader, validloader, epochs)


bless_train_predict.checkpoint_save(model,train_data, arch1, epochs, dropout, hidden_layer, lr)

