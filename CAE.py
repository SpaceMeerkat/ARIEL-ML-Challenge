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
        self.lc1 = torch.nn.Linear(55*int(300/5),2048)
        self.lc2 = torch.nn.Linear(2048,1024)
        self.lc3 = torch.nn.Linear(1024,512)
        self.lc4 = torch.nn.Linear(512,256)
        self.lc5 = torch.nn.Linear(256,128)
        self.lc6 = torch.nn.Linear(129,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
        y = y.view(int(y.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        x = self.lc5(x)
        x = self.relu(x)
        x = torch.cat((x,y),1)
        x = self.lc6(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x,y)
        return output  
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class CovNet(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.conv1 = torch.nn.Conv1d(1,32,3,padding=1)
        self.conv2 = torch.nn.Conv1d(32,64,3,padding=1)
        self.conv3 = torch.nn.Conv1d(64,128,3,padding=1)
        self.mp = torch.nn.AvgPool1d(2)
        self.mp2 = torch.nn.AvgPool1d(3)
        self.lc1 = torch.nn.Linear(128*275,2048)
        self.lc2 = torch.nn.Linear(2048,1024)
        self.lc3 = torch.nn.Linear(1024,512)
        self.lc4 = torch.nn.Linear(512,256)
        self.lc5 = torch.nn.Linear(261,self.nodes) #257,self.nodes
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x):
        x = x.view(int(x.size()[0]),1,-1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x)
        y = y.unsqueeze(1)
        output = torch.cat((output,y),dim=1)
        output = self.lc5(output)
        return output  
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#    

class CovNet2(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.conv1 = torch.nn.Conv1d(1,32,3,padding=1)
        self.conv2 = torch.nn.Conv1d(32,64,3,padding=1)
        self.conv3 = torch.nn.Conv1d(64,128,3,padding=1)
        self.ap = torch.nn.AvgPool1d(2)
        self.ap2 = torch.nn.AvgPool1d(3)
        self.lc1 = torch.nn.Linear(128*275,2048)
        self.lc2 = torch.nn.Linear(2048,1024)
        self.lc3 = torch.nn.Linear(1024,512)
        self.lc4 = torch.nn.Linear(512,256)
        self.lc5 = torch.nn.Linear(256,55)
        self.lc6 = torch.nn.Linear(60,self.nodes)
       # self.lc7 = torch.nn.Linear(self.nodes,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x):
        x = x.view(int(x.size()[0]),1,-1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.ap(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.ap(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.ap2(x)
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        return x
        
    def forward(self,x,y):
        output = self.encoder(x)
        #y = y.unsqueeze(1)
        output = self.lc5(output)
        output = self.relu(output)
        output = torch.cat((output,y),dim=1)
        output = self.lc6(output)
        #output = self.relu(output)
        #output = self.lc6(output)
        #output = self.relu(output)
        #output = self.lc7(output)
        return output  
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class LinearNet_3(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*int(300/3),2048)
        self.lc2 = torch.nn.Linear(2048,2048)
        self.lc3 = torch.nn.Linear(2048,1024)
        self.lc4 = torch.nn.Linear(1024,512)
        self.lc5 = torch.nn.Linear(512,256)
        self.lc6 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
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
        
class LinearNet_5(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*int(300/5),2048)
        self.lc2 = torch.nn.Linear(2049,2048)
        self.lc3 = torch.nn.Linear(2048,1024)
        self.lc4 = torch.nn.Linear(1024,512)
        self.lc5 = torch.nn.Linear(512,256)
        self.lc6 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
        y = y.view(int(y.size()[0]),-1)
        x = self.lc1(x)
        x = torch.cat((x,y),1)
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
               
class LinearNet_10(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*int(300/10),2048)
        self.lc2 = torch.nn.Linear(2048,2048)
        self.lc3 = torch.nn.Linear(2048,1024)
        self.lc4 = torch.nn.Linear(1024,512)
        self.lc5 = torch.nn.Linear(512,256)
        self.lc6 = torch.nn.Linear(256,self.nodes)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
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
        
class AE_5(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
        self.lc1 = torch.nn.Linear(55*int(300/5),2048)
        self.lc2 = torch.nn.Linear(2048,2048)
        self.lc3 = torch.nn.Linear(2048,1024)
        self.lc4 = torch.nn.Linear(1024,512)
        self.lc5 = torch.nn.Linear(512,self.nodes)

        self.lc8 = torch.nn.Linear(self.nodes,512)
        self.lc9 = torch.nn.Linear(512,1024)
        self.lc10 = torch.nn.Linear(1024,2048)
        self.lc11 = torch.nn.Linear(2048,2048)
        self.lc12 = torch.nn.Linear(2048,55*int(300/5))
        self.relu = torch.nn.ReLU()
        
        self.lc14 = torch.nn.Linear(192,128)
        self.lc15 = torch.nn.Linear(128,55)
        
    def encoder(self,x,y):
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        x = self.relu(x)
        x = self.lc3(x)
        x = self.relu(x)
        x = self.lc4(x)
        x = self.relu(x)
        x = self.lc5(x)
        return x
    
    def decoder(self,x):
        x = self.lc8(x)
        x = self.relu(x)
        x = self.lc9(x)
        x = self.relu(x)
        x = self.lc10(x)
        x = self.relu(x)
        x = self.lc11(x)
        x = self.relu(x)
        x = self.lc12(x)
        x = x.view(int(x.size()[0]),55,int(300/5))
        return x
    
    def branch(self,x):
        x = self.lc14(x)
        x = self.relu(x)
        x = self.lc15(x)
        return x
        
    def forward(self,x,y):
        encoded = self.encoder(x,y)
        decoded = self.decoder(encoded)
        branched = self.branch(encoded[:,:192])
        return decoded, branched

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#        
