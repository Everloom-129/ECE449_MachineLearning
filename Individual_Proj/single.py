import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch import nn
import os
import random
import time
import torch
import random
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

def load_coil20_dataset(base_directory, transform, test_split=0.2):
    # Initialize lists to store image paths and labels
    image_paths = []
    labels = []
    
    # Navigate through the subdirectories and collect image paths and labels
    for i in range(1, 21):  # Assuming categories are labeled from 01 to 20
        directory = os.path.join(base_directory, f'{i:02}')  # Format subdirectory path
        images_in_dir = glob.glob(f"{directory}/*.png")
        image_paths.extend(images_in_dir)
        labels.extend([i] * len(images_in_dir))  # Label is the category number

    # Shuffle the dataset to ensure randomness
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)
    
    # Splitting the dataset into train and test
    split_idx = int(len(image_paths) * (1 - test_split))
    train_paths, test_paths = image_paths[:split_idx], image_paths[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Create the dataset objects
    train_dataset = COIL20Dataset(train_paths, train_labels, transform)
    test_dataset = COIL20Dataset(test_paths, test_labels, transform)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# Custom Dataset class for COIL-20 dataset.
class COIL20Dataset(Dataset):
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label  # Return the image-label pair

    def __len__(self):
        return len(self.images)  # Return the total number of images

# Neural Network for image classification.
class Classify_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the network layers.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 20)  # Assuming 20 classes in the dataset
        )

    def forward(self, x):
        # Implement the forward pass.
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.linear_relu_stack(x)
        return x

# Parse custom command-line arguments.
def get_config():
    parser = argparse.ArgumentParser(description='COIL-20 image classification parameters.')
    # Add arguments to the parser.
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    # Add other necessary arguments.
    
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    return args

# Main class for training and testing the network.
class MyNet:
    # Constructor for MyNet.
    def __init__(self, args,train_loader,test_loader):
        self.args = args
        self.train_loader = train_loader 
        self.test_loader = test_loader  
        self.model = Classify_NN().to(args.device)  # Initialize the model and send to device.
        self.criterion = nn.CrossEntropyLoss()  # Define the loss function.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Define the optimizer.

    def _train(self, dataloader):
        # Implement the training code.
        self.model.train()  # Set the model to training mode.
        total_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(self.args.device), target.to(self.args.device)
            self.optimizer.zero_grad()  # Zero the parameter gradients.
            output = self.model(data)  # Forward pass.
            loss = self.criterion(output, target)  # Compute the loss.
            loss.backward()  # Backpropagation.
            self.optimizer.step()  # Update the weights.
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

    def _test(self, dataloader):
        # Implement the testing code.
        self.model.eval()  # Set the model to evaluation mode.
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

    def run(self):
        # Implement the run function for the training and testing loop.
        for epoch in range(self.args.epochs):
            train_loss, train_accuracy = self._train(self.train_loader)
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}')
            test_loss, test_accuracy = self._test(self.test_loader)
            print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    config = get_config()  # Get the command line arguments.
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])
    net = MyNet(config,load_coil20_dataset('coil-20-proc/',transform))  # Initialize MyNet with the provided arguments.
    net.run()  # Start the training and testing process.
