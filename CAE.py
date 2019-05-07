# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:22:48 2019

@author: c1307135
"""

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

import torch

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class LinearNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*60,2048)
        self.lc2 = torch.nn.Linear(2048,2048)
        self.lc3 = torch.nn.Linear(2048,1024)
        self.lc4 = torch.nn.Linear(1024,512)
        self.lc5 = torch.nn.Linear(512,256)
        self.lc6 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
#        y = y.view(int(y.size()[0]),-1)
        x = self.lc1(x)
#        x = torch.cat((x,y),1)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        x = self.lc5(x)
        x = self.relu(x)
        x = self.lc6(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x,y)
        return output  
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#        
        
class DeepLinearNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*60,2048)
        self.lc2 = torch.nn.Linear(2048,2048)
        self.lc3 = torch.nn.Linear(2048,2048)
        self.lc4 = torch.nn.Linear(2048,2048)
        self.lc5 = torch.nn.Linear(2048,1024)
        self.lc6 = torch.nn.Linear(1024,512)
        self.lc7 = torch.nn.Linear(512,256)
        self.lc8 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        self.droput = torch.nn.Dropout(0.5)
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
#        y = y.view(int(y.size()[0]),-1)
        x = self.lc1(x)
#        x = torch.cat((x,y),1)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.droput(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.droput(x)
        x = self.lc4(x)
        x = self.relu(x)
        x = self.droput(x)
        x = self.lc5(x)
        x = self.relu(x)
        x = self.lc6(x)
        x = self.relu(x)
        x = self.lc7(x)
        x = self.relu(x)
        x = self.lc8(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x,y)
        return output  
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
        
class ConvNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.conv1 = torch.nn.Conv2d(1,8,3,padding=1) 
        self.conv2 = torch.nn.Conv2d(8,8,3,padding=1)
        self.conv3 = torch.nn.Conv2d(8,16,3,padding=1)
        self.conv4 = torch.nn.Conv2d(16,16,3,padding=1)
        self.lc1 = torch.nn.Linear(16*15*15,2560)
        self.lc2 = torch.nn.Linear(2560,2560)
        self.lc3 = torch.nn.Linear(2560,1280)
        self.lc4 = torch.nn.Linear(1280,55)
        self.mp = torch.nn.MaxPool2d(2,return_indices=False)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x):
        x = self.conv1(x) # 8,60,60  
        x = self.relu(x)
        x = self.conv2(x) # 8,60,60
        x = self.relu(x)
        x = self.mp(x) # 8,30,30        
        x = self.conv3(x) # 16,30,30
        x = self.relu(x)
        x = self.conv4(x) # 16,30,30
        x = self.relu(x)
        x = self.mp(x) # 16,15,15
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x) 
        x = self.lc2(x) 
        x = self.lc3(x) 
        x = self.lc4(x) 
        return x
        
    def forward(self,x):
        output = self.encoder(x)
        return output        

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#        