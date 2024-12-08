import torch
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from glob import glob
from os import path
from typing import Optional
import math

import zipfile


class ImagesDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width: int = 100,
            height: int = 100,
            dtype: Optional[type] = None
    ):
        self.image_filepaths = sorted(path.abspath(f) for f in glob(path.join(image_dir, "*.jpg")))
        class_filepath = [path.abspath(f) for f in glob(path.join(image_dir, "*.csv"))][0]
        self.filenames_classnames, self.classnames_to_ids = ImagesDataset.load_classnames(class_filepath)
        if width < 100 or height < 100:
            raise ValueError('width and height must be greater than or equal 100')
        self.width = width
        self.height = height
        self.dtype = dtype

    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()
        classnames_to_ids = {}
        for index, classname in enumerate(classnames):
            classnames_to_ids[classname] = index
        return filenames_classnames, classnames_to_ids

    def __getitem__(self, index):
        with Image.open(self.image_filepaths[index]) as im:
            image = np.array(im, dtype=self.dtype)
        image = to_grayscale(image)
        resized_image, _ = prepare_image(image, self.width, self.height, 0, 0, 32)
        resized_image = torch.tensor(resized_image, dtype=torch.float32)/255.0
        classname = self.filenames_classnames[index][1]
        classid = self.classnames_to_ids[classname]
        return resized_image, classid, classname, self.image_filepaths[index]

    def __len__(self):
        return len(self.image_filepaths)
    
def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int):
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("image must have shape (1, H, W)")
    if width < 32 or height < 32 or size < 32:
        raise ValueError("width/height/size must be >= 32")
    if x < 0 or (x + size) > width:
        raise ValueError(f"x={x} and size={size} do not fit into the resized image width={width}")
    if y < 0 or (y + size) > height:
        raise ValueError(f"y={y} and size={size} do not fit into the resized image height={height}")
    
    image = image.copy()

    if image.shape[1] > height:
        image = image[:, (image.shape[1] - height) // 2: (image.shape[1] - height) // 2 + height, :]
    else: 
        image = np.pad(image, ((0, 0), ((height - image.shape[1])//2, math.ceil((height - image.shape[1])/2)), (0, 0)), mode='edge')
    
    if image.shape[2] > width:
        image = image[:, :, (image.shape[2] - width) // 2: (image.shape[2] - width) // 2 + width]
    else:
        image = np.pad(image, ((0, 0), (0, 0), ((width - image.shape[2])//2, math.ceil((width - image.shape[2])/2))), mode='edge')

    subarea = image[:, y:y + size, x:x + size]
    return image, subarea
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def _init_(self):
        super(MyCNN, self)._init_()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # New layers
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjusted the input dimensions
        self.fc2 = nn.Linear(256, 20)  # Assuming 20 classes for classification
        
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # Convolutional layers with batch normalization and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # New convolutional layers
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 3 * 3)  # Adjusted the input dimensions
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = MyCNN()




parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
architecture = "architecture.py"
trained_model = "model.pth"

exec(open(architecture).read())
model.load_state_dict(torch.load(r"C:\Users\omark\OneDrive - Johannes Kepler Universität Linz\Johannes Kepler Universität\Programming for AI II\Assignment 7 Project\MISC\SAAD tes\model.pth", map_location=torch.device(device)))
model.eval()

location_test_data = r"C:\Users\omark\OneDrive - Johannes Kepler Universität Linz\Johannes Kepler Universität\Programming for AI II\Assignment 7 Project\training_data\training_data"
test_dataset = ImagesDataset(location_test_data, 100, 100, int)

test_dl = DataLoader(dataset=test_dataset, shuffle=False, batch_size=32)
correct = 0
total = 0

with torch.no_grad():
    for X, y, _, _ in test_dl:         
        y_pred = model(X)
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(correct/total)
