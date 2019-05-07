# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:46:43 2019

@author: c1307135
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
from data_loader import FITSCubeDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_path = '/home/anubis/c1307135/ARIEL/TEST_FLUX/'
save_path = '/home/anubis/c1307135/ARIEL/features/'
batch_size=640

total = int(29360/batch_size)
blind_test = False
if blind_test == True:
    total = int(62900/batch_size)
    data_path = '/home/anubis/c1307135/ARIEL/noisy_test_fits/'


model_path = '/home/anubis/c1307135/ARIEL/models/'
model = torch.load(model_path+'ARIEL_LinearNet_Slow_Epoch_95.pt').cpu()
model.train(False)
        
_filenames = FITSCubeDataset(data_path).make_dataset()

test_loader = DataLoader(dataset=FITSCubeDataset(data_path,return_filename=True),
                              batch_size=batch_size,num_workers=16,shuffle=True)

all_names = []
all_features = []
all_targets = []

for idx, (batch, radius, target, names) in tqdm(enumerate(test_loader),total=total):
    _features = model.encoder(batch.float(),radius.float()).detach().numpy()
    all_names.append(names)
    all_features.append(_features)
    if blind_test == False:
        all_targets.append(target)
        
all_features = np.vstack(all_features)
all_names = np.hstack(all_names)

if blind_test == False:
    all_targets = np.vstack(all_targets)
    average_err = np.mean(np.abs(all_targets.reshape(-1)-all_features.reshape(-1)))
    plt.figure()
    plt.scatter(all_targets.reshape(-1),all_features.reshape(-1),c='k',s=0.2)
    plt.plot([0,1],[0,1],'r--',alpha=0.5)
    plt.text(0.1,0.7,str('%.6f' % (average_err)),bbox=dict(facecolor='grey',alpha=0.5))
    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    plt.savefig('/home/corona/c1307135/Holy_Grail/test_images/Linear_Slow.png')
    
if blind_test == True:
    df = pd.DataFrame(all_features)
    df.insert(loc=0,column='NAME',value=all_names)
    df = df.sort_values(by=['NAME'])
    df = df.drop(['NAME'],axis=1)
    df.to_pickle(save_path+'ARIEL_features_noisy_test.pkl')
    df = pd.read_pickle(save_path+'ARIEL_features_noisy_test.pkl')
    df.to_csv('/home/corona/c1307135/Holy_Grail/test_images/results/params_test.txt',
              sep=' ',header=False,index=False)

