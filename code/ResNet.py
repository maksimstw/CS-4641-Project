import torch.nn as nn
import torch.optim as optim
import torch.hub
from tqdm import tqdm
from utils import RoadSignDataset, check_accuracy, save_checkpoint
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 182
learning_rate = 3e-4
batch_size = 32
num_epochs = 10
load_model = False

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
# model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', num_classes=10).to(device=device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Load model?
def load_checkpoint(checkpoint):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if load_model:
    load_checkpoint(torch.load("checkpoint/checkpoint.pth.tar"))

# Start training
for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=f'checkpoints/resnet18_epoch {epoch}')

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
    check_accuracy(train_loader, model, "train")
    check_accuracy(val_loader, model, "val")

print("Checking final accuracy on training set...")
check_accuracy(train_loader, model, "final train")

print("Checking final accuracy on validation set...")
check_accuracy(val_loader, model, "final val")

checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint, filename=f'checkpoints/resnet18_final')
