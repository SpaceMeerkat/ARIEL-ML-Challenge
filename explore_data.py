# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:53:17 2019

@author: c1307135
"""

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

import os
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class ARIEL:
    
    def filename_collector(self, initial_path):
        f = []
        for filename in tqdm(glob.glob(initial_path+'*.txt')):
            f.append(os.path.basename(filename))
        return f
    
    def data_loader(self,file_path,bin_width):
        _content = np.loadtxt(file_path,delimiter='\t')
        _content = pd.DataFrame(_content.T); _content[_content.iloc[:,:]>1]=np.nan;
        _step = bin_width; _dt = np.arange(0,301,_step);
        _df_content = _content.groupby(pd.cut(_content.index,_dt)).mean()
        _df_content.index = _dt[:-1] + (_step/2)
        return _df_content
        
    def target_loader(self,file_path):
        _content = np.loadtxt(file_path,delimiter='\t')
        _content = pd.DataFrame(_content.T)
        return _content
    
    def lightcurve(self,fluxes,save_path):
        plt.figure()
        plt.scatter(fluxes.index,fluxes,c='k',s=0.5,linestyle='-')
        plt.xlabel('t'); plt.ylabel('Relative brightness');
        plt.tight_layout()
        plt.savefig(save_path+'test.png')

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

#_path = "/home/anubis/c1307135/ARIEL/noisy_test/"
#_filenames = ARIEL().filename_collector(initial_path=_path)
#_low_noise = []
#for i in _filenames:
#    i = i[:-4]
#    if i[-2:]=='01' and i[-5:-3]=='01':
#        _low_noise.append(i+'.txt')
#_file_name = _low_noise[1]
#_df = ARIEL().data_loader(file_path=_path+_file_name, bin_width=5)
#_save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
#ARIEL().lightcurve(fluxes=_df[1],save_path=_save_path)

_target_path = '/home/anubis/c1307135/ARIEL/params_train/'
_target_files = ARIEL().filename_collector(initial_path=_target_path)
_targets = ARIEL().target_loader(_target_path+_target_files[0])
print(_targets)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#