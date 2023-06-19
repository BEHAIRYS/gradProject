
# import the necessary packages
from torch import nn
class CNN(nn.Module):
    def __init__(self):     
        super(CNN,self).__init__()
        self.conva=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=0,stride=1)
        self.convb=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=5,padding=0,stride=1)
        self.convc=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=5,padding=0,stride=1)
        self.convd=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=5,padding=0,stride=1)
        self.conve=nn.Conv2d(in_channels=6,out_channels=6,kernel_size=5,padding=0,stride=1)
        self.relu=nn.ReLU()
    def forward(self,x):
      out=self.conva(x)
      out=self.convb(out)
      out=self.convc(out)
      out=self.convd(out)
      out=self.conve(out)
      return out

