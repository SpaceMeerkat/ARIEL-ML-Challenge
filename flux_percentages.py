#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:24:54 2019

@author: c1307135
"""

import os
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from astropy.io import fits

class flux_percentiles:
    
    def __init__(self,data_path,save_path):
        self.load_path = data_path
        self.save_path = save_path
        
    def filename_collector(self):
        _f = []
        for filename in tqdm(glob.glob(self.load_path+'*.fits')):
            _f.append(filename)
        return _f
       
    def open_file(self,filename):
        _file = fits.open(filename)
        _data = _file[1].data.astype(float) 
        _percentiles = np.nanpercentile(_data,1,axis=1)
        return _percentiles
        
    def pool_session(self,processors=None):
        if processors is None: processors = 1
        _f = self.filename_collector()
        pool = mp.Pool(processes = processors)
        _results = list(tqdm(pool.imap(self.open_file,_f),total=len(_f)))
        _results = [x for x in _results if str(x) != 'nan']
        _results = np.vstack(_results)
        _results = np.median(_results,axis=0)
        print(_results.shape)
        _df = pd.DataFrame(_results)
        print(_df.shape)
        _df.to_pickle(os.path.join(self.save_path,'percentiles_no_nan_final.pkl'))
        return _results
       
path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/TRAIN/'
save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
_percentiles = flux_percentiles(path,save_path).pool_session(processors=14)
print(_percentiles)       
