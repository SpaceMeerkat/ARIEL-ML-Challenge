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
import multiprocessing as mp

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class ARIEL_data:
    
    def __init__(self,bin_width=None,percentile_path=None):
        self.bin_width = bin_width
        self.percentile_path = percentile_path
    
    def filename_collector(self, initial_path):
        f = []
        for filename in tqdm(glob.glob(initial_path+'*.txt')):
            f.append(os.path.basename(filename))
        return f
        
    def filename_collector_long(self, initial_path):
        f = []
        for filename in tqdm(glob.glob(initial_path+'*.txt')):
            f.append(filename)
        return f
    
    def data_loader(self,file_path,bin_width):
        _content = np.loadtxt(file_path,delimiter='\t')
        _content = pd.DataFrame(_content.T); _content[_content.iloc[:,:]>1]=np.nan;
        _step = bin_width; _dt = np.arange(0,301,_step);
        _df_content = _content.groupby(pd.cut(_content.index,_dt)).mean()
        _df_content.index = _dt[:-1] + (_step/2)
        return _df_content
    
    def lightcurve(self,fluxes,save_path):
        plt.figure()
        plt.scatter(fluxes.index,fluxes,c='k',s=0.5,linestyle='-')
        plt.xlabel('t'); plt.ylabel('Relative brightness');
        plt.tight_layout()
        plt.savefig(save_path+'test.png')
        
    def make_image(self,file_path,save_path,plot=None):
        _df = self.data_loader(file_path,self.bin_width)
        _data = _df.values.astype(float).T
        if self.percentile_path is not None:
            _percentiles = pd.read_pickle(self.percentile_path).values
            _data -= _percentiles
            _data /= (1-_percentiles)
        _data[_data!=_data] = 1
        if plot is not None:
            plt.figure(); plt.imshow(_data); plt.colorbar();
            plt.gca().invert_yaxis()
            plt.savefig(save_path+os.path.basename(file_path)+'.png')
            
    def get_all_data(self,file_path):
        _df_content = self.data_loader(file_path,self.bin_width).values.T
        return _df_content
        
    def pool_session(self,file_paths,processes,save_path):
        pool = mp.Pool(processes = processes)
        results = list(tqdm(pool.imap(self.get_all_data,file_paths),total=len(file_paths)))
        results = [x for x in results if str(x) != 'nan']
        df = pd.DataFrame(np.hstack(results))
        _percentiles = np.nanpercentile(df.values,1,axis=1)
        _df_percentiles = pd.DataFrame(_percentiles.T)
        _df_percentiles.columns = ['PERCENTILE']
        _df_percentiles.to_pickle(save_path+'channel_percentiles.pkl')

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class ARIEL_targets:
    
    def filename_collector(self, initial_path):
        f = []
        for filename in tqdm(glob.glob(initial_path+'*.txt')):
            f.append(filename)
        return f

    def target_loader(self,file_path):
        _content = np.loadtxt(file_path,delimiter='\t')
        _content = pd.DataFrame(_content.T)
        _content.columns = ['LABEL']
        return _content     
        
    def target_distribution(self,filenames,wavelength,save_path=None):
        _targets = []
        for _filename in tqdm(filenames,total=len(filenames)):
            _content = self.target_loader(_filename)
            _target = _content.iloc[wavelength].values.astype(float)
            _targets.append(_target)
        if  save_path is not None:
            _df_targets = pd.DataFrame(_targets)
            _df_targets.columns = ['LABEL']
            _df_targets.to_pickle(save_path+'targets_wavelength_'+str(wavelength)+'.pkl')
        return _targets
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
        
        
class ARIEL_dataset_creator:
    
    def __init__(self,data_path,target_path,save_path,bin_width,percentile_path,percentiles=None):
        self.data_path = data_path
        self.target_path = target_path
        self.save_path = save_path
        self.bin_width = bin_width
        self.percentile_path = percentile_path
        self.percentiles = percentiles
    
    def percentile_loader(self):
        self.percentiles = pd.read_pickle(self.percentile_path).values
        
    def filename_collector(self):
        f = []
        for filename in tqdm(glob.glob(self.data_path+'*.txt')):
            f.append(os.path.basename(filename))
        return f
    
    def image_maker(self,file_path):
        if self.percentiles is None:
            self.percentile_loader()
        _content = np.loadtxt(self.data_path+file_path,delimiter='\t').T
        _content = pd.DataFrame(_content); _content[_content.iloc[:,:]>1]=np.nan;
        _step = self.bin_width; _dt = np.arange(0,301,_step);
        _df_content = _content.groupby(pd.cut(_content.index,_dt)).mean()
        _df_content.index = _dt[:-1] + (_step/2)
        _data = _df_content.values.astype(float).T
        _data -= self.percentiles; _data /= (1-self.percentiles);
        _data[_data!=_data] = 1
        return _data.T
        
    def target_loader(self,file_path):
        _content = np.loadtxt(self.target_path+file_path,delimiter='\t')
        return _content 
            
    def make_all_data(self,_file):
        _image = self.image_maker(_file)
        _target = self.target_loader(_file)
        _grouped_data = np.vstack([_target,_image])
        _df_grouped_data = pd.DataFrame(_grouped_data.T)
        _df_grouped_data.to_pickle(self.save_path+_file[:-4]+'.pkl')
        return
        
    def pool_session(self,processors):
        _file_directories = self.filename_collector()
        pool = mp.Pool(processes = processors)
        results = list(tqdm(pool.imap(self.make_all_data,_file_directories),total=len(_file_directories)))
        results = [x for x in results if str(x) != 'nan']
            
   
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    