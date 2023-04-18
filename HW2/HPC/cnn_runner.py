#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from skimage import io
from skimage import color
# pytorch:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
# save output:
import pickle
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None): 
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0]) # df 1st col is file name
        x = io.imread(img_name, as_gray=True).astype('float32')  # read img with io
        x = (x - x.mean())/x.std()   # standard normalization
        y = self.data_frame["Class"][idx]  # get outcome
        if self.transform:
            x = self.transform(x).float()  # make sure output is float32
        else:              # if transformation -> dimension will be 1*n*n
            x = x[None,:]  # broadcasting if didn't transform
        sample = {'x': x, 'y': y}  # return dictionary of image and corresponding class
        return sample
class ReLU1(nn.Module):    # nn.ReLU has issue in my pytorch version 
    def forward(self, x):  # ref: https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256?page=2
        return F.relu6(x * 6.0) / 6.0  # turn relu6 to relu1
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__() 
        # block 1:
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),  # use padding=2 to avoid mismatch
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),          # follow the paper arxiv: 1512.03385 do batchnorm 
            ReLU1(),                    # before activation
            nn.Conv2d(8, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            ReLU1()
        )
        # use method 2. in ResNet paper, 1*1 convolution with strides, to downsample: 
        self.downsample1 = nn.Conv2d(1,16,1,stride=4, bias=False)  
        # block 2:      
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            ReLU1(), 
            nn.Conv2d(32, 64, 5, stride=1, padding=2),  # for residual
            nn.BatchNorm2d(64),
            ReLU1()
        )
        self.downsample2 = nn.Conv2d(16,64,1,stride=4, bias=False)
        # block 3:
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            ReLU1(),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),  # for residual
            nn.BatchNorm2d(64),
            ReLU1()
        )  
        self.downsample3 = nn.Conv2d(64,64,1,stride=4, bias=False) 
        # block 4:
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            ReLU1()
        )
        # FC:
        self.fc1 = nn.Linear(128*4*4, 128)
        self.fc2 = nn.Linear(128, 3)
    def forward(self, x):
        residual = self.downsample1(x) # to match the size after convolution & pooling
        x = self.block1(x)  
        x += residual     
        residual = self.downsample2(x)
        x = self.block2(x)
        x += residual
        residual = self.downsample3(x)
        x = self.block3(x)
        x += residual
        x = self.block4(x)
        x = x.view(-1, 128*4*4)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)           # didn't do softmax
        return x
def conv_train(train_data, test_data, model, optimizer, batch_size, num_epochs, device, verbose=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()   # multi-class
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    n_total_steps = len(train_loader)  # len(train_sample) / num_epochs
    train_loss = []
    test_loss = []
    y_true_ord = torch.tensor([]).to(device)    # to save true outcome with same order as prediction
    y_prob_ord = torch.tensor([]).to(device)    # predicted prob
    for epoch in range(num_epochs):
        train_loss_acc = test_loss_acc = 0
        for i, x_y in enumerate(train_loader):
            x_train = x_y["x"].to(device)
            y_train = x_y["y"].to(device)
            y_train = y_train.long()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)  # record learning rate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_acc += loss.item()
            if (i+1) % verbose == 0:
                print(f"Epoch[{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
        train_loss.append(train_loss_acc/len(train_loader))
        with torch.no_grad():   
            for i, x_y in enumerate(test_loader):
                x_test = x_y["x"].to(device)
                y_test = x_y["y"].to(device)
                y_test = y_test.long() # .unsqueeze(1)
                outputs = model(x_test)#.squeeze()
                loss = criterion(outputs, y_test)
                y_true_ord = torch.cat([y_true_ord, y_test], dim=0)  # true outcome ordered
                y_prob_ord = torch.cat([y_prob_ord, outputs], dim=0) # predicted prob ordered
                test_loss_acc += loss.item()   
            test_loss.append(test_loss_acc/len(test_loader))
    y_true = y_true_ord.detach().cpu().numpy().astype("int")
    y_prob = y_prob_ord.detach().cpu().numpy()
    results = {"train_loss":train_loss, "test_loss":test_loss, "y_true":y_true, "y_prob":y_prob}
    return results
def save_py(obj, location):
    """
    TODO: to save a python object
    :param obj: object name
    :param location: location to save the file
    """
    if location[-4:] != '.pkl': location += '.pkl'  # add file extension
    savefile = open(f'{location}', 'wb')
    pickle.dump(obj, savefile)
    savefile.close()
#  code to run:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device to use: {device}")
train_sample = ChestXrayDataset("/scratch/ct2840/DL_course/HW2/dataset/HW2_trainSet_new.csv", 
"/scratch/ct2840/DL_course/HW2/images_tar/images/", transform=None)
test_sample = ChestXrayDataset("/scratch/ct2840/DL_course/HW2/dataset/HW2_validationSet_new.csv", 
"/scratch/ct2840/DL_course/HW2/images_tar/images/", transform=None)
batch_size = 48   
num_epochs = 30     
verbose = 24  
learning_rate = 0.005  
model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print(f"parameters set up finish: batch size={batch_size}, epochs={num_epochs}, optimizer={optimizer}")
torch.manual_seed(1)
results = conv_train(train_data=train_sample, test_data=test_sample, 
                     model = model, optimizer = optimizer, 
                     batch_size=batch_size, num_epochs = num_epochs,
                     device=device, verbose=verbose
                                  )
save_py(results, "output/cnn_Q49_result_03252pm")




