
# import the necessary packages
from torch import nn
class CNN(nn.Module):
    def __init__(self):     
        super(CNN,self).__init__()
        self.conva=nn.Conv2d(in_channels=1,out_channels=5,kernel_size=6,padding=0,stride=1)
        self.maxa=nn.MaxPool2d(2,stride=1,padding=0)
        self.convb=nn.Conv2d(in_channels=5,out_channels=16,kernel_size=6,padding=0,stride=1)
        self.maxb=nn.MaxPool2d(2,stride=1,padding=0)
        self.fca=nn.Linear(in_features=4096,out_features=120)
        self.fcb=nn.Linear(in_features=120*1*1,out_features=84)
        self.fcc=nn.Linear(in_features=84*1*1,out_features=10)
        self.relu=nn.ReLU()
    def forward(self,x):
      out=self.conva(x)
      out=self.maxa(out)
      out=self.convb(out)
      out=self.maxb(out)
      out= out.view(out.shape[0], -1)
      out=self.fca(out)
      out=self.fcb(out)
      out=self.fcc(out)
      return out

