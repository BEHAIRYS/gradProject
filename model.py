
# import the necessary packages
from torch import nn
class CNN(nn.Module):
    def __init__(self):     
        super(CNN,self).__init__()
        self.conva=nn.Conv2d(in_channels=3,out_channels=5,kernel_size=6,padding=0,stride=2)
        self.maxa=nn.MaxPool2d(2,stride=2,padding=0)
        self.convb=nn.Conv2d(in_channels=5,out_channels=5,kernel_size=2,padding=0,stride=1)
        self.avgb=nn.AvgPool2d(2,stride=1,padding=0)
        self.fca=nn.Linear(in_features=125*1*1,out_features=5)
        self.relu=nn.ReLU()
    def forward(self,x):
      out=self.conva(x)
      out=self.maxa(out)
      out=self.convb(out)
      out=self.avga(out)
      out= out.view(out.shape[0], -1)
      out=self.fca(out)
      return out

