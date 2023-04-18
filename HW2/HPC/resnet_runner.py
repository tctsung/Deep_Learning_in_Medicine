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
import torch.optim as optim
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
# save output:
import pickle

class ChestXrayDataset_ResNet(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        # self.data_frame = load_data_and_get_class(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = image[:,:,0]
        image=np.repeat(image[None,...],3,axis=0)
        image = torch.tensor(image).float()
        image_class = torch.tensor(self.data_frame["Class"][idx])
        if self.transform:
            image = self.transform(image) 
        sample = {'x': image, 'y': image_class}
        return sample
def resnet_train(train_loader, test_loader, model, optimizer, batch_size, num_epochs, device="cpu", verbose=5):
    print(f"Device to use: {device}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
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
                y_test = y_test.long()  
                outputs = model(x_test)
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
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(896),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_sample = ChestXrayDataset_ResNet("/scratch/ct2840/DL_course/HW2/dataset/HW2_trainSet_new.csv", 
                                "/scratch/ct2840/DL_course/HW2/images_tar/images/", transform=train_transform)
validate_sample = ChestXrayDataset_ResNet("/scratch/ct2840/DL_course/HW2/dataset/HW2_validationSet_new.csv", 
                               "/scratch/ct2840/DL_course/HW2/images_tar/images/", transform=validation_transform)
batch_size = 48
train_loader = DataLoader(dataset=train_sample, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=validate_sample, batch_size=batch_size, shuffle=False)   
num_epochs = 20     
verbose = 24  
learning_rate = 0.001 
model = models.resnet18(weights=None)  # no pre-trained weights
num_ftrs = model.fc.in_features  
model.fc = nn.Linear(num_ftrs, 3)      # 3 classes
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print(f"parameters set up finish: batch size={batch_size}, epochs={num_epochs}, optimizer={optimizer}")
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = resnet_train(train_loader=train_loader, test_loader=test_loader, model=model, 
             optimizer=optimizer, batch_size=batch_size, num_epochs=num_epochs, 
             device=device, verbose=verbose)
save_py(results, "output/resnet_result_gpu")




