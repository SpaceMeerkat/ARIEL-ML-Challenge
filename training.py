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
from CAE import LinearNet, CovNet2
from data_loader import FITSCubeDataset
from plotting_routine import epoch_plotter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt

torch.cuda.benchmark=True
torch.cuda.fastest =True

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

def learning_rate(initial_lr, epoch):
    """Sets the learning rate to the initial LR decayed by a factor of 10 every
    N epochs"""
    lr = initial_lr * (0.9 ** (epoch// 1))
    return lr 
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   

def train(model:torch.nn.Module,data_path,test_path,perc_path,num_epochs,
          batch_size,loss_function,initial_lr,train_total,test_total,
          save_model_path):
    
    _image_path = '/home/corona/c1307135/Holy_Grail/training_files/'    
    
    train_loader = DataLoader(dataset=FITSCubeDataset(data_path,perc_path),
                              batch_size=batch_size,num_workers=16,shuffle=True)
                              
    test_loader = DataLoader(dataset=FITSCubeDataset(test_path,perc_path),
                              batch_size=batch_size,num_workers=16,shuffle=True)
                              
    device = torch.device("cuda")
    _model = model.to(device).to(torch.float)
       
    print('-'*73)
    print('MODEL INFORMATION')
    print(_model)
    print('-'*73)
       
    train_epoch, training_loss, training_std = [],[],[]
    test_epoch, test_loss, test_std = [],[],[]
    
    print('BEGINNING GPU SLAVERY :)')
    print('-'*73)
              
    for epoch in range(num_epochs+1):
        
        train_epoch.append(epoch)
        print("Epoch {0} of {1}" .format( (epoch+1), num_epochs))
        optim = torch.optim.Adam(model.parameters(),learning_rate(initial_lr,epoch))
        running_training_loss = []    
        model.train(True) 
        for idx, (batch, _radii, target, _logg, _mass, _kmag, _period,_nspot, _nphot) in tqdm(enumerate(train_loader),total=train_total):
            _batch = batch.to(device).to(torch.float)
            _radii = np.vstack([_radii,_logg,_mass,_kmag,_period]).T
            _radii = torch.tensor(_radii)
            _radii = _radii.to(device).to(torch.float)
            _target = target.to(device).to(torch.float)
            _prediction = _model(_batch,_radii)
            _loss = loss_function(_prediction,_target)
            optim.zero_grad(); _loss.backward(); optim.step();
            running_training_loss.append(_loss.detach().cpu())
        training_loss.append(np.mean(running_training_loss))
        training_std.append(np.std(running_training_loss))
        print("Mean training loss: %.8f" % training_loss[epoch]) 
        
        if epoch % 5 == 0:
            test_epoch.append(epoch)
            running_test_loss = []
            model.train(False)
            for idx, (batch, _radii, target,_logg, _mass, _kmag, _period,_nspot, _nphot) in tqdm(enumerate(test_loader),total=test_total):  
                _batch = batch.to(device).to(torch.float)
                _radii = np.vstack([_radii,_logg,_mass,_kmag,_period]).T
                _radii = torch.tensor(_radii)
                _radii = _radii.to(device).to(torch.float)
                _target = target.to(device).to(torch.float)
                _prediction = _model(_batch,_radii)
                _loss = loss_function(_prediction,_target)
                running_test_loss.append(_loss.detach().cpu()) 
            test_loss.append(np.mean(running_test_loss))
            test_std.append(np.std(running_test_loss))
            print('-'*73,"\n Mean test loss: %.8f" % np.mean(running_test_loss),
                  "\n Mean test std: %f" % np.std(running_test_loss))
            print('-'*73)
        if epoch % 5 == 0:
            epoch_plotter(train_epoch,training_loss,training_std,
                 test_epoch,test_loss,test_std,_image_path+'CovNet2_meta_test2.png').plotter()
            torch.save(model.state_dict(),save_model_path+'CovNet2_meta_test2.pt')
    model.train(False)
     
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#       
    
data_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/TRAIN/'
test_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/TEST/'
perc_path = '/home/corona/c1307135/Holy_Grail/test_images/percentiles_no_nan_final.pkl' #percentiles_no_nan.pkl
save_model_path = '/home/anubis/c1307135/ARIEL/models/'
batch_size = 128 #256
total_train = int(117440/batch_size)
total_test = int(29360/batch_size)
train(model=CovNet2(55),data_path=data_path,test_path=test_path,perc_path=perc_path,
      num_epochs=75,batch_size=batch_size,
      loss_function=torch.nn.MSELoss(),initial_lr=1e-3,train_total=total_train,test_total=total_test,
      save_model_path=save_model_path)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   


#            if idx % 10 == 0:
#                _batch = batch.to(device).to(torch.float)
#                b = _batch.detach().cpu().numpy()
#                plt.figure()
#                plt.imshow(b[0,0,:,:])
#                plt.savefig(_image_path+'test.png')
#                plt.close('all')
