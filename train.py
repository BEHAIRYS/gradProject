
# import the necessary packages
import os
import torch
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.optim import Adadelta, Adam
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch import nn
import pandas as pd
from torch.optim import Adadelta
from model import CNN
from torch.utils.data import Dataset, random_split, DataLoader

#initiallization
LR = 0.001
BATCH_SIZE = 256
EPOCHS = 100
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.15
TEST_SPLIT = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
#size=dataset.__len__()
#train_dataset, val_dataset, test_dataset = random_split(dataset, [size*TRAIN_SPLIT, size*VAL_SPLIT, size*TEST_SPLIT],generator=torch.Generator().manual_seed(42))
train_dataset =  mnist.MNIST(root='./train', train=True, transform=ToTensor(),download=True)
test_dataset = mnist.MNIST(root='./', train=False, transform=ToTensor(),download=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
opt = Adam(model.parameters(), lr=1e-1)
lossFn = nn.CrossEntropyLoss()

for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (x, y) in train_dataloader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
    model.eval()
    print(trainCorrect)

torch.save(model)
    
