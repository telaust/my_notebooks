import numpy as np

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

from torchvision import datasets

class CNN(nn.Module):
    def __init__(self, in_channels=1, n_classe=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1,),
                padding=(1, 1))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(
                in_channels=8,
                out_channels=18,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1))
        self.fc1 = nn.Linear(18*6*6, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x



device = 'cpu'

in_channels = 1
batch_size = 128
n_epochs = 3
n_classes = 10

train_dataset = datasets.MNIST(root="dataset/",train=True, transform=T.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=T.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

model = CNN()


crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 1e-2)

losses = []
for epoch in range(n_epochs):
    for data, targets in tqdm(train_loader):
        
        scores = model(data)
        loss = crit(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    print(f'epoch: {epoch+1}, mean loss={sum(losses)/len(losses)}')


def check_accuracy_on_test(test_loader, model):
    model.eval()
    num_corrects = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            # use cuda if you can
            x = x.to(device)
            y = y.to(device)
    
            out = model(x)
            _, preds = out.max(1)
            n_corrects += (preds == y).sum()
            n_samples += preds.size(0)
            
        acc = float(num_corrects) / float(num_samples) * 100
        print(f"accuracy on test data ={acc}")

plt.plot(losses)
plt.ylabel("loss")
plt.xlabel("num of iterations")
plt.grid(True)
plt.show()

