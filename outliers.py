#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 11:13:13 2019

@author: jamesdawson
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
from astropy.io import fits

data_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/TEST/'
perc_path = '/home/corona/c1307135/Holy_Grail/test_images/percentiles_no_nan.pkl'
save_path = '/home/corona/c1307135/Holy_Grail/outliers_images/'

d = fits.open(data_path+'1846_06_05.fits')

data = d[1].data.astype(float) 

plt.figure()
plt.imshow(data)
plt.colorbar()
plt.savefig(save_path+'test.png')

plt.figure()
plt.plot(np.median(data,axis=1),'k.')
plt.savefig(save_path+'test2.png')


'''
test_loader = DataLoader(dataset=FITSCubeDataset(data_path,perc_path=perc_path,return_filename=True),
                              batch_size=1,num_workers=16,shuffle=False)
'''
