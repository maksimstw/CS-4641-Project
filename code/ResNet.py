import torch
import torchvision
import time
import os
import copy
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import RoadSignDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, recall_score, precision_score

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 182
learning_rate = 3e-4
batch_size = 128
num_epochs = 10

# Data Transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Load Data
dataset = RoadSignDataset(csv_file='data/labels.csv', root_dir='data/train_images',
                          transform=data_transforms)

train_set, val_set = torch.utils.data.random_split(dataset, [36833, 9229])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

# Model
model = models.resnet18(pretrained=True).to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    print("Checking accuracy...")

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}')

        f1 = f1_score(y_true=y.cpu(), y_pred=predictions.cpu(), average='macro')
        recall = recall_score(y_true=y.cpu(), y_pred=predictions.cpu(), average='macro')
        precision = precision_score(y_true=y.cpu(), y_pred=predictions.cpu(), average='macro')

        print(f'Got macro f1 {f1} with recall {recall} and precision {precision}')

    model.train()


for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

    print(f'Epoch {epoch}: cost is {sum(losses) / len(losses)}')
    check_accuracy(train_loader, model)

print("Checking final accuracy on training set...")
check_accuracy(train_loader, model)

print("Checking final accuracy on validation set...")
check_accuracy(val_loader, model)
