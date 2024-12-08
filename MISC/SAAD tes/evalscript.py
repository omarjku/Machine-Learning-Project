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
from dataset import ImagesDataset
import zipfile
from architecture import model


parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
architecture = [r"C:\Users\omark\Downloads\architecture.py"]
trained_model = [r"C:\Users\omark\Downloads\model.pth"]

exec(open(architecture).read())
model.load_state_dict(torch.load(trained_model, map_location=torch.device(device)))
model.eval()

location_test_data = r"C:\Users\omark\OneDrive - Johannes Kepler Universität Linz\Johannes Kepler Universität\Programming for AI II\Assignment 7 Project\training_data\training_data"
test_dataset = ImagesDataset(location_test_data, 100, 100, int)
indices = np.arange(len(test_dataset))
test_indices = indices[int(len(indices) * (4 / 5)):]
test_set = torch.utils.data.Subset(test_dataset, test_indices)
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
