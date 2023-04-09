
# import the necessary packages
import os
import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch import nn
from torch.testing._internal.common_utils import args
import pandas as pd
from torch.optim import Adadelta
from torch.utils.data import Dataset, random_split, DataLoader
class CNN(nn.Module):
    def __init__(self):     
        super(CNN,self).__init__()
        self.conva=nn.Conv2d(in_channels=0,out_channels=3,kernel_size=328,padding=0,stride=1)
        self.maxa=nn.MaxPool2d(328,stride=1,padding=0)
        self.avgb=nn.AvgPool2d(328,stride=1,padding=0)
        self.relu=nn.ReLU()
    def forward(self,x):
      out=self.conva(x)
      out=self.maxa(x)
      out=self.avga(x)
      return out

