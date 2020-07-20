# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:33:39 2019

@author: c1307135
"""
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

from explore_data import ARIEL_data, ARIEL_dataset_creator
from astropy.io import fits
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pickle

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

lc_plotter = False
if lc_plotter == True:
    _path = "/home/anubis/c1307135/ARIEL/noisy_test/"
    _filenames = ARIEL_data().filename_collector(initial_path=_path)
    _low_noise = []
    for i in _filenames:
        i = i[:-4]
        if i[-2:]=='01' and i[-5:-3]=='01':
            _low_noise.append(i+'.txt')
    for j in range(10):
        _file_name = _low_noise[j]
        _df = ARIEL_data().data_loader(file_path=_path+_file_name, bin_width=5)
        _save_path = '/home/corona/c1307135/Holy_Grail/test_images/lightcurves/'
        ARIEL_data().lightcurve(fluxes=_df[0],save_path=_save_path,name=str(j))
    
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

pool_session = True
if pool_session == True:
    _path = "/home/anubis/c1307135/ARIEL/noisy_test/"
    _target_path = '/home/anubis/c1307135/ARIEL/params_train/'
    _save_path = '/home/anubis/c1307135/ARIEL/flux_files/NO_NAN/BLIND/'
    ARIEL_dataset_creator(data_path=_path,target_path=None,save_path=_save_path,
                          bin_width=5).pool_session(processors=14)
                              
def pooler(_file):
    f = fits.open(_file)
    r = float(f[0].header['RSTAR'])
    return r
    
radii_explorer = False
if radii_explorer == True:
    _path = "/home/anubis/c1307135/ARIEL/TEST_FLUX/"
    _save_path = '/home/anubis/c1307135/ARIEL/'
    _filenames = ARIEL_data().filename_collector_long(initial_path=_path,extension='.fits')
    pool = mp.Pool(processes = 14)
    results = list(tqdm(pool.imap(pooler,_filenames),total=len(_filenames)))
    results = [x for x in results if str(x) != 'nan']
    results = np.array(results)
    results = results**2
    _min = results.min()
    _max = results.max()
    scalers = [_min,_max]
    with open(_save_path+'scalers.pkl','wb') as handle:
        pickle.dump(scalers, handle)
    


    
