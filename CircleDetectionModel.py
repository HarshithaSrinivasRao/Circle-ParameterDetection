import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleDetection(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CircleDetection,self).__init__()
           

        self.conv=nn.Conv2d(in_channels=in_channels,kernel_size=3, out_channels=out_channels,stride=1,padding=1)
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        self.relu=nn.ReLU()
    

    def forward(self,x):
        out=self.conv(x)
        out=self.bn(out)
        out=self.relu(out)

        return out


class Detection(nn.Module):
    def __init__(self,batch_size=10):
        super(Detection,self).__init__()

        self.batch_size=batch_size
        
        self.block1=CircleDetection(in_channels=1,out_channels=64)
        self.block2=CircleDetection(in_channels=64,out_channels=64)
        self.block3=CircleDetection(in_channels=64,out_channels=64)

        self.maxpool1=nn.MaxPool2d(2,2)

        self.block4=CircleDetection(in_channels=64,out_channels=128)
        self.block5=CircleDetection(in_channels=128,out_channels=128)
        self.block6=CircleDetection(in_channels=128,out_channels=128)
        self.block7=CircleDetection(in_channels=128,out_channels=128)

        self.maxpool2=nn.MaxPool2d(2,2)

        self.block8=CircleDetection(in_channels=128,out_channels=256)
        self.block9=CircleDetection(in_channels=256,out_channels=256)
        self.block10=CircleDetection(in_channels=256,out_channels=256)
        self.block11=CircleDetection(in_channels=256,out_channels=256)

        

        self.network=nn.Sequential(self.block1,self.block2,self.block3,self.maxpool1,self.block4,self.block5,self.block6,self.block7,self.maxpool2,
                                   self.block8,self.block9,self.block10,self.block11)

        self.fc1=nn.Linear(in_features=256*50*50,out_features=256)
        self.fc2=nn.Linear(in_features=256,out_features=64)
        self.fc3=nn.Linear(in_features=64,out_features=3)
        

    def forward(self,x):
        #print('len of x',len(x),'size of x',x.size())
        x=self.network(x)
        #print('x after detection net',x.size())
        x=x.view(x.size(0),-1)
        #print('x after view',x.size())
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        #print('final output',x)
        return x





         
