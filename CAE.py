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

class Autoencoder(torch.nn.Module):
               
    def __init__(self,nodes):
        super().__init__()
        self.nodes = nodes
      
        self.conv1 = torch.nn.Conv2d(1,8,3,padding=1) 
        self.conv2 = torch.nn.Conv2d(8,8,3,padding=1)
        self.conv3 = torch.nn.Conv2d(8,16,3,padding=1)
        self.conv4 = torch.nn.Conv2d(16,16,3,padding=1)
        self.lc1 = torch.nn.Linear(16*15*15,self.nodes)
        self.lc2 = torch.nn.Linear(self.nodes,16*15*15)
        self.trans1 = torch.nn.ConvTranspose2d(16,16,3,padding=1)
        self.trans2 = torch.nn.ConvTranspose2d(16,8,3,padding=1)
        self.trans3 = torch.nn.ConvTranspose2d(8,8,3,padding=1)
        self.trans4 = torch.nn.ConvTranspose2d(8,1,3,padding=1)
        self.mp = torch.nn.MaxPool2d(2,return_indices=True)
        self.up = torch.nn.MaxUnpool2d(2)
        self.relu = torch.nn.ReLU()
        
    def encoder(self,x):
        x = self.conv1(x) # 8,60,60  
        x = self.relu(x)
        x = self.conv2(x) # 8,60,60
        x = self.relu(x)
        s1 = x.size()
        x, ind1 = self.mp(x) # 8,30,30        
        x = self.conv3(x) # 16,30,30
        x = self.relu(x)
        x = self.conv4(x) # 16,30,30
        x = self.relu(x)
        s2 = x.size()
        x, ind2 = self.mp(x) # 16,15,15
        x = x.view(int(x.size()[0]),-1)
        x = self.lc1(x)  
        return x,ind1,s1,ind2,s2
        
    def decoder(self,x,ind1,s1,ind2,s2):
        x = self.lc2(x)
        x = x.view(int(x.size()[0]),16,15,15)
        x = self.up(x, ind2, output_size=s2) 
        x = self.relu(x)
        x = self.trans1(x)
        x = self.relu(x)
        x = self.trans2(x)
        x = self.up(x, ind1, output_size =s1)
        x = self.relu(x)
        x = self.trans3(x)
        x = self.relu(x)
        x = self.trans4(x)     
        return x        
        
    def forward(self,x):
        x,ind1,s1,ind2,s2 = self.encoder(x)
        output = self.decoder(x,ind1,s1,ind2,s2)
        return output
        
        
        