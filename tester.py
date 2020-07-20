# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:46:43 2019

@author: c1307135
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from data_loader import FITSCubeDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from CAE import CovNet2, LinearNet_5

data_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/TEST/'

save_path = '/home/anubis/c1307135/ARIEL/features/'
batch_size=640

total = int(29360/batch_size)
total2 = 29360
blind_test = True
if blind_test == True:
    total = int(62900/batch_size)
    total2 = 62900
    data_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/BLIND/'

model_path = '/home/anubis/c1307135/ARIEL/models/'
perc_path = '/home/corona/c1307135/Holy_Grail/test_images/percentiles_no_nan.pkl'

#model2 = LinearNet_5(55)
#model2.load_state_dict(torch.load(model_path+'ARIEL_LinearNet_nonan_r_75.pt'))

model = CovNet2(55)
model.load_state_dict(torch.load(model_path+'CovNet2_meta_test.pt'))

print('-'*73)
print('MODEL INFORMATION: \n',model)
print('-'*73)

model = model.cpu()
model.train(False)
       
_filenames = FITSCubeDataset(data_path).make_dataset()

test_loader = DataLoader(dataset=FITSCubeDataset(data_path,perc_path=perc_path,return_filename=True),
                              batch_size=batch_size,num_workers=16,shuffle=False)

all_names, all_features, all_targets = [],[],[]

print('-'*73)
print('BEGINNING TESTING PHASE')

for idx, (batch, radius, target,_l,_m,_k,_p,_nspot, _nphot, names) in tqdm(enumerate(test_loader),total=total):
    _radius=np.vstack([radius,_l,_m,_k,_p]).T
    _radius=torch.tensor(_radius)
    _features = model(batch.float(),_radius.float()).detach().numpy()
    all_names.append(names)
    all_features.append(_features)
    if blind_test == False:
        all_targets.append(target)
print('-'*73)
        
all_features = np.vstack(all_features)
all_names = np.hstack(all_names)
print(all_features.shape)

#=============================================================================#

second_wind = False
if second_wind == True:
    indices = []
    for k in tqdm(range(all_features.shape[0])):
        if len(np.where(all_features[k,:]>0.175)[0])>0:
            indices.append(k)
    print(len(indices))
    
    test_loader2 = DataLoader(dataset=FITSCubeDataset(data_path,perc_path=perc_path,return_filename=True),
                                  batch_size=total2,num_workers=16,shuffle=False)
    
    for idx, (batch, radius, target, names) in tqdm(enumerate(test_loader2),total=1):  
        batch = batch[indices,:]; radius = radius[indices]
        _features = model2(batch.float(),radius.float()).detach().numpy()
        for j in range(len(indices)):
            all_features[indices[j],:] = _features[j]
    print('-'*73)

#=============================================================================#
    
if blind_test == False:
    all_targets = np.vstack(all_targets)
    print(all_targets.shape)
    average_err = np.mean(np.abs(all_targets.reshape(-1)-all_features.reshape(-1)))
    
    suspects = []
    for i in range(all_targets.shape[0]):
            if len(np.where((all_targets[i,:]>0.08)&(all_targets[i,:]<0.1))[0])>(0.5*(all_targets.shape[1])):
                    suspects.append(i)  
                    
    suspects2 = []
    for i in suspects:
            if len(np.where(all_features[i,:]<0.06)[0]) > (0.5*(all_features.shape[1])):
                    suspects2.append(int(i))
                    
   # print(len(suspects2))
   # print(all_names[suspects2[0]])
                        
    plt.figure()
    plt.scatter(all_targets.reshape(-1),all_features.reshape(-1),c='k',s=0.2)
    plt.scatter(all_targets[suspects2].reshape(-1),all_features[suspects2].reshape(-1),c='r',s=0.2)
    plt.plot([0,1],[0,1],'r--',alpha=0.5)
    plt.xlabel('True');  plt.ylabel('Predicted')
    plt.text(0.1,0.7,str('%.6f' % (average_err)),bbox=dict(facecolor='grey',alpha=0.5))
    print(average_err)
    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    plt.savefig('/home/corona/c1307135/Holy_Grail/test_images/CovNet2_meta_test.png')
    
if blind_test == True:
    df = pd.DataFrame(all_features)
    df.insert(loc=0,column='NAME',value=all_names)
    df = df.sort_values(by=['NAME'])
    df = df.drop(['NAME'],axis=1)
    df.to_pickle(save_path+'ARIEL_features_noisy_test.pkl')
    df = pd.read_pickle(save_path+'ARIEL_features_noisy_test.pkl')
    df.to_csv('/home/corona/c1307135/Holy_Grail/test_images/results/params_test_CovNet2_meta_test.txt',
              sep=' ',header=False,index=False)
              

#=============================================================================#
    
"""
for idx, (batch, radius, target, names) in tqdm(enumerate(test_loader2),total=1):
    for i in tqdm(range(batch.shape[0])):
        _features = model(batch[i,:,:,:].float(),radius[i].float().unsqueeze(0)).detach().numpy()
        all_features.append(_features)
        if blind_test == False:
            all_targets.append(target[i])
    all_names.append(names)
print('-'*73)

"""
