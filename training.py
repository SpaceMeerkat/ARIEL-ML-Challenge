# -*- coding: utf-8 -*-
"""
Created on Wed May  1 13:29:33 2019

@author: c1307135
"""

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from CAE import Autoencoder
from data_loader import FITSCubeDataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.benchmark=True
torch.cuda.fastest =True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

def learning_rate(initial_lr, epoch):
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    2 epochs"""
    lr = initial_lr * (0.5 ** (epoch // 2))
    return lr

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

def train(model: torch.nn.Module,data_path,num_epochs,
          batch_size,loss_function,initial_lr,total):
    
    train_loader = DataLoader(dataset=FITSCubeDataset(data_path),
                              batch_size=batch_size,num_workers=16,shuffle=True)
                              
    device = torch.device("cuda")
    _model = model.to(device).to(torch.float)
    
    training_loss, training_std = [],[]
              
    for epoch in range(num_epochs):
        print("Epoch {0} of {1}" .format( (epoch+1), num_epochs))
        optim = torch.optim.Adam(model.parameters(), learning_rate(initial_lr,epoch))
        running_training_loss = []    
        model.train(True) 
        for idx, (batch, target) in tqdm(enumerate(train_loader),total=total):
            _batch = batch.to(device).to(torch.float)
            _prediction = _model(_batch)
            _loss = loss_function(_prediction,_batch)
            optim.zero_grad(); _loss.backward(); optim.step();
            running_training_loss.append(_loss.detach().cpu())
        training_loss.append(np.mean(running_training_loss))
        training_std.append(np.std(running_training_loss))
        print("Mean training loss: %f" % training_loss[epoch]) 
    model.train(False)
            
            
data_path = '/home/anubis/c1307135/ARIEL/TRAIN_FLUX/'
batch_size=64
total = int(146800/batch_size)
train(model=Autoencoder(3),data_path=data_path,num_epochs=20,batch_size=batch_size,
      loss_function=torch.nn.MSELoss(),initial_lr=1e-3,total=total)





