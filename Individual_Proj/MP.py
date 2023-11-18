# # Project Description
# 
# In this project, you need to solve an image classification task using coil-20-proc dataset.
# 
# This dataset consists of 1,440 grayscale images of 20 objects (72 images per object). 
# - Half of each category is for training and half for testing. (e.g. 36 images for training and another 36 images for testing).
# 

# ## The COIL-20 (Columbia Object Image Library) dataset
# a collection of grayscale images of 20 different objects. The "proc" in "COIL-20-proc" indicates that the images in this dataset have been preprocessed. Each object has been photographed on a motorized turntable against a black background to provide a 360-degree view. The objects are varied, including toys, household items, and office supplies, to provide a range of shapes and complexities.
# 
# 
# The preprocessing typically involves normalizing the images in terms of scale and orientation, and the background is often removed to isolate the object of interest. This makes the dataset particularly useful for computer vision tasks like object recognition and classification, where consistency in the input data can significantly improve the performance of machine learning models.
# 
# The COIL-20 dataset is commonly used in academia for teaching and research in the fields of computer vision and machine learning, as it presents a controlled environment that allows for the evaluation of algorithms and techniques.

# # 1. Complete the custom dataset, and dataloader.

# %%
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch import nn

class COIL20Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
    def __len__(self):
        return len(self.images)


def load_coil20_dataset(directory, transform, test_split=0.2):
    # Get all the image file paths
    image_paths = glob.glob(f"{directory}/*.png") # Assuming the images are in PNG format
    labels = [int(path.split('_')[1]) for path in image_paths]  # Extracting labels from file names

    # Splitting the dataset into train and test
    split_idx = int(len(image_paths) * (1 - test_split))
    train_paths, test_paths = image_paths[:split_idx], image_paths[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Creating the dataset objects
    train_dataset = COIL20Dataset(train_paths, train_labels, transform)
    test_dataset = COIL20Dataset(test_paths, test_labels, transform)

    # Creating the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# Example transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# the COIL-20 images are stored in a directory as below:
# - coil-20-proc/
#       - 01/obj1__0.png, obj1__1.png, ...
#       - 02/...
#       - ...
#       - 20/....
train_loader, test_loader = load_coil20_dataset('coil-20-proc/01/', transform)



# %%
import torch
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %% [markdown]
# # 2. Implementing a Neural Network
# Fill in the code to complete the custom model, you can refer to https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# %%

class Classify_NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear

        )
    def forward(self, x):
    # you should complete forward function

        return x

# %% [markdown]
# # 3. Customize some arguments
# Fill in the code to complete the custom arguments

# %%
import os
import sys
import time
import torch
import random
import argparse
from datetime import datetime


def get_config():
    parser = argparse.ArgumentParser()

    # you should complete arguments
    parser.add_argument()

    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args

# %% [markdown]
# # 4. Train and Test the model.
# Fill in the code to complete the training and testing, and print the results and losses. You can refer to https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html and https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# %%
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm


class MyNet:
    def __init__(self, args, logger):
        self.args = args

        # Operate the method
        self.Mymodel = Classify_NN()
        self.Mymodel.to(args.device)
        self._print_args()

    def _train(self, dataloader, criterion, optimizer):
    # you should complete the training code, optimizer and loss function
        
        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
    # you should complete the testing code

        return test_loss / n_test, n_correct / n_test

    def run(self):
    # you should complete run function for training and testing the model, and print the results and losses.


if __name__ == '__main__':
    args = get_config()
    net = MyNet(args)
    net.run()
