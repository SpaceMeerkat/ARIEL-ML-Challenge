# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:33:39 2019

@author: c1307135
"""
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

from explore_data import ARIEL_data, ARIEL_targets, ARIEL_dataset_creator

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
    _file_name = _low_noise[1]
    _df = ARIEL_data().data_loader(file_path=_path+_file_name, bin_width=5)
    print(_df.shape)
    _save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
    ARIEL_data().lightcurve(fluxes=_df[0],save_path=_save_path)

data_image = False
if data_image == True:
    _path = "/home/anubis/c1307135/ARIEL/noisy_test/"
    _save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
    _filenames = ARIEL_data().filename_collector(initial_path=_path)   
    ARIEL_data().make_image(_path+_filenames[0],bin_width=10,save_path=_save_path,plot=True)
    
get_all_data = False
if get_all_data == True:
    _path = "/home/anubis/c1307135/ARIEL/noisy_train/"
    _save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
    _filenames = ARIEL_data().filename_collector_long(initial_path=_path)   
    ARIEL_data(bin_width=10).pool_session(_filenames,processes=14,save_path=_save_path)
    
data_image_norm = False
if data_image_norm == True:
    _path = "/home/anubis/c1307135/ARIEL/noisy_train/"
    _perc_path = '/home/corona/c1307135/Holy_Grail/test_images/channel_percentiles.pkl'
    _save_path = '/home/corona/c1307135/Holy_Grail/test_images/normed/'
    _filenames = ARIEL_data().filename_collector_long(initial_path=_path) 
    ARIEL_data(bin_width=5,percentile_path=_perc_path).make_image(_filenames[2],save_path=_save_path,plot=True)    

target_grabber = False
if target_grabber == True:
    _target_path = '/home/anubis/c1307135/ARIEL/params_train/'
    _target_files = ARIEL_targets().filename_collector(initial_path=_target_path)
    _targets = ARIEL_targets().target_loader(_target_path+_target_files[0])
    print(_targets.iloc[54].values)
    
target_dists = False
if target_dists == True:
    _target_path = '/home/anubis/c1307135/ARIEL/params_train/'
    _save_path = '/home/corona/c1307135/Holy_Grail/test_images/'
    _target_files = ARIEL_targets().filename_collector(initial_path=_target_path)
    _distribution_1 = ARIEL_targets().target_distribution(_target_files,54,save_path=_save_path) 

_path = "/home/anubis/c1307135/ARIEL/noisy_train/"
_target_path = '/home/anubis/c1307135/ARIEL/params_train/'
_save_path = '/home/anubis/c1307135/ARIEL/TRAIN_FLUX/'
_perc_path = '/home/corona/c1307135/Holy_Grail/test_images/channel_percentiles.pkl'
ARIEL_dataset_creator(data_path=_path,target_path=_target_path,save_path=_save_path,
                      bin_width=5,percentile_path=_perc_path).pool_session(processors=14)
    